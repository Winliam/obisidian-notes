"""
股价网格交易策略推演脚本 — 5221策略
买入比例分配：首次买入50%，之后每跌10%分别买入20%、20%、10%
共4次买入，资金100%用完
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import numpy as np
from matplotlib.gridspec import GridSpec

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import matplotlib
matplotlib.use('Agg')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'DejaVu Sans']
plt.rcParams['font.serif']      = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10


def verify_font_config():
    import matplotlib.font_manager as fm
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    chinese_fonts   = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi']
    found = [f for f in chinese_fonts if f in available_fonts]
    if found:
        plt.rcParams['font.sans-serif'].remove(found[0])
        plt.rcParams['font.sans-serif'].insert(0, found[0])
        print(f"✓ 检测到中文字体: {found[0]}")
    else:
        print("⚠ 警告: 未检测到系统中文字体，将使用默认字体")


verify_font_config()


# ═══════════════════════════════════════════════════════════════════════════════
# 模拟器
# ═══════════════════════════════════════════════════════════════════════════════

class GridTradingSimulator:
    """
    5221网格交易策略模拟器

    与511111策略的关键区别：
      grid_invest_ratios 是一个列表，每次网格买入使用不同比例
      5221 → [0.20, 0.20, 0.10]（第1/2/3次网格买入）
    """

    def __init__(self, config: Dict):
        self.total_amount        = config.get('total_amount', 100000)
        self.initial_price       = config.get('initial_price', 10.0)
        self.first_buy_ratio     = config.get('first_buy_ratio', 0.5)
        self.grid_drop_ratio     = config.get('grid_drop_ratio', 0.1)
        # ★ 核心差异：变长列表，每层网格对应自己的投入比例
        self.grid_invest_ratios  = config.get('grid_invest_ratios', [0.2, 0.2, 0.1])
        self.profit_target_ratio = config.get('profit_target_ratio', 0.3)
        self.stock_name          = config.get('stock_name', '某股票')
        self._validate_config()

        # 状态变量
        self.buy_records         = []
        self.total_invested      = 0.0
        self.total_shares        = 0.0
        self.avg_cost            = 0.0
        self.price_history       = []
        self.cost_history        = []
        self.target_price_history = []

    def _validate_config(self):
        assert self.total_amount > 0,                     "总投入资金必须大于0"
        assert self.initial_price > 0,                    "初始价格必须大于0"
        assert 0 < self.first_buy_ratio <= 1,             "首次买入比例必须在(0,1]"
        assert 0 < self.grid_drop_ratio <= 1,             "网格下跌比例必须在(0,1]"
        assert all(0 < r <= 1 for r in self.grid_invest_ratios), \
                                                          "每层网格投入比例必须在(0,1]"
        total_ratio = self.first_buy_ratio + sum(self.grid_invest_ratios)
        assert abs(total_ratio - 1.0) < 1e-6,            \
            f"各层比例之和应为1，当前为{total_ratio:.4f}"
        assert 0 < self.profit_target_ratio <= 2,         "目标盈利比例必须在(0,2]"

    # ─── 主模拟循环 ────────────────────────────────────────────────────────────
    def simulate_trading(self, price_list: List[float]) -> Dict:
        # 重置
        self.buy_records          = []
        self.total_invested       = 0.0
        self.total_shares         = 0.0
        self.avg_cost             = 0.0
        self.price_history        = []
        self.cost_history         = []
        self.target_price_history = []

        sell_price = None
        sell_date  = None

        for date_idx, current_price in enumerate(price_list):
            self.price_history.append(current_price)

            # ── 步骤1：执行买入 ──────────────────────────────────────────
            if date_idx == 0:
                self._buy(current_price, date_idx, is_first=True)
            else:
                first_buy_price  = self.buy_records[0]['price']
                price_drop_ratio = (first_buy_price - current_price) / first_buy_price

                if price_drop_ratio > 0:
                    # +1e-9 修正浮点截断误差（0.3/0.1=2.9999...会被int截为2）
                    grid_level        = int(price_drop_ratio / self.grid_drop_ratio + 1e-9)
                    current_buy_count = len(self.buy_records)   # 已买次数（含首次）
                    remaining         = self.total_amount - self.total_invested
                    max_grids         = len(self.grid_invest_ratios)

                    # 触发了新层级 且 该层级在策略范围内 且 还有余额
                    if (grid_level > (current_buy_count - 1)
                            and grid_level <= max_grids
                            and remaining > 0):
                        self._buy(current_price, date_idx)

            # ── 步骤2：记录历史（买入之后再记录，保证当日数值正确） ────
            self.cost_history.append(self.avg_cost)
            self.target_price_history.append(
                self.avg_cost * (1 + self.profit_target_ratio) if self.avg_cost > 0 else 0
            )

            # ── 步骤3：检查止盈（首日跳过） ─────────────────────────────
            if date_idx > 0 and self.total_shares > 0:
                if self._check_sell_condition(current_price):
                    # 精确止盈价，避免离散价格序列带来的误差
                    sell_price = self.avg_cost * (1 + self.profit_target_ratio)
                    sell_date  = date_idx
                    break

        return self._generate_report(sell_price, sell_date)

    # ─── 买入执行 ──────────────────────────────────────────────────────────────
    def _buy(self, price: float, date_idx: int, is_first: bool = False):
        if is_first:
            ratio = self.first_buy_ratio
        else:
            # 当前是第几次网格买入（0-indexed）
            grid_idx = len(self.buy_records) - 1   # buy_records已含首次
            ratio    = self.grid_invest_ratios[grid_idx]

        amount    = self.total_amount * ratio
        remaining = self.total_amount - self.total_invested
        if amount > remaining:
            amount = remaining

        shares              = amount / price
        self.total_invested += amount
        self.total_shares   += shares
        self.avg_cost        = self.total_invested / self.total_shares

        self.buy_records.append({
            'date_idx':      date_idx,
            'price':         price,
            'amount':        amount,
            'ratio':         ratio,
            'shares':        shares,
            'total_invested': self.total_invested,
            'total_shares':  self.total_shares,
            'avg_cost':      self.avg_cost,
            'grid_level':    len(self.buy_records),   # 记录前的长度即层级编号
        })

    def _check_sell_condition(self, current_price: float) -> bool:
        if self.total_shares == 0:
            return False
        return (current_price - self.avg_cost) / self.avg_cost >= self.profit_target_ratio

    # ─── 报告生成 ──────────────────────────────────────────────────────────────
    def _generate_report(self, sell_price: float = None, sell_date: int = None) -> Dict:
        if sell_price is not None:
            sell_amount   = self.total_shares * sell_price
            profit_amount = sell_amount - self.total_invested
            profit_ratio  = profit_amount / self.total_invested
            reach_target  = True
        else:
            sell_amount = profit_amount = profit_ratio = 0
            reach_target = False

        first_buy_price      = self.buy_records[0]['price']
        cost_reduction       = (first_buy_price - self.avg_cost) / first_buy_price if len(self.buy_records) > 1 else 0
        cost_reduction_amount = first_buy_price - self.avg_cost if len(self.buy_records) > 1 else 0

        layers_info = [{
            '网格层级': r['grid_level'],
            '买入比例': f"{r['ratio']:.0%}",
            '买入价格': f"{r['price']:.4f}",
            '购买股数': f"{r['shares']:.4f}",
            '投入金额': f"{r['amount']:.2f}",
            '累计投入': f"{r['total_invested']:.2f}",
            '平均成本': f"{r['avg_cost']:.4f}",
        } for r in self.buy_records]

        return {
            '股票':       self.stock_name,
            '总投入资金': f"¥{self.total_invested:.2f}",
            '买入次数':   len(self.buy_records),
            '首次买入价格': f"¥{first_buy_price:.4f}",
            '平均成本价': f"¥{self.avg_cost:.4f}",
            '成本降低':   f"{cost_reduction:.2%} (¥{cost_reduction_amount:.4f})",
            '累计持股':   f"{self.total_shares:.4f}股",
            '分层投入':   layers_info,
            '卖出信息': {
                '卖出价格': f"¥{sell_price:.4f}" if sell_price is not None else "未卖出",
                '卖出日期': sell_date if sell_date is not None else "N/A",
                '卖出金额': f"¥{sell_amount:.2f}",
                '盈利金额': f"¥{profit_amount:.2f}",
                '盈利率':   f"{profit_ratio:.2%}",
                '达到目标': "是" if reach_target else "否",
            }
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 场景分析
# ═══════════════════════════════════════════════════════════════════════════════

def scenario_analysis(config: Dict, test_scenarios: List[Dict]):
    print("\n\n" + "="*80)
    print("   5221策略场景分析：不同下跌深度下的收益情况")
    print("="*80)

    simulators = []
    summary    = []

    for scenario in test_scenarios:
        sim    = GridTradingSimulator(config)
        result = sim.simulate_trading(scenario['prices'])
        simulators.append((scenario['name'], sim, result))
        sell   = result['卖出信息']
        summary.append({
            '场景':     scenario['name'],
            '最低价':   f"¥{min(scenario['prices']):.4f}",
            '平均成本': result['平均成本价'],
            '止盈价':   sell['卖出价格'],
            '盈利金额': sell['盈利金额'],
            '盈利率':   sell['盈利率'],
            '达到目标': sell['达到目标'],
        })

    print("\n" + pd.DataFrame(summary).to_string(index=False))
    print("\n" + "="*80)
    return simulators


# ═══════════════════════════════════════════════════════════════════════════════
# 图表1：每个场景的详细走势图
# ═══════════════════════════════════════════════════════════════════════════════

def create_scenario_visualization(simulators: List[Tuple], config: Dict, out_dir: str):
    n   = len(simulators)
    fig, axes = plt.subplots(n, 2, figsize=(16, 5 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle('5221网格策略 — 各场景价格走势与成本分析',
                 fontsize=15, fontweight='bold', y=1.01)

    C_PRICE  = '#1f77b4'
    C_COST   = '#ff7f0e'
    C_TARGET = '#2ca02c'

    for row, (name, sim, result) in enumerate(simulators):
        ax_l = axes[row, 0]
        x    = range(len(sim.price_history))

        ax_l.plot(x, sim.price_history,        marker='o', lw=2, label='股价',
                  color=C_PRICE,  markersize=6)
        ax_l.plot(x, sim.cost_history,         marker='s', lw=2, label='平均成本价',
                  color=C_COST,   markersize=5)
        ax_l.plot(x, sim.target_price_history, marker='^', lw=2, linestyle='--',
                  label='目标止盈价', color=C_TARGET, markersize=5)

        # 买入标记（红色向下三角）
        for rec in sim.buy_records:
            ax_l.scatter(rec['date_idx'], rec['price'],
                         s=220, marker='v', color='red', zorder=5,
                         edgecolors='darkred', linewidths=1.5,
                         label=f"买入@¥{rec['price']:.2f} ({rec['ratio']:.0%})")

        # 卖出标记（绿色向上三角）
        sell = result['卖出信息']
        if sell['卖出价格'] != '未卖出' and sell['卖出日期'] != 'N/A':
            sp = float(sell['卖出价格'].replace('¥', ''))
            ax_l.scatter(sell['卖出日期'], sp,
                         s=220, marker='^', color='limegreen', zorder=5,
                         edgecolors='darkgreen', linewidths=1.5,
                         label=f"卖出@¥{sp:.4f}")

        ax_l.set_title(f'场景{row+1}: {name}\n价格走势与成本分析',
                       fontsize=11, fontweight='bold', pad=8)
        ax_l.set_xlabel('时间周期', fontsize=10)
        ax_l.set_ylabel('价格（元）', fontsize=10)
        ax_l.legend(loc='best', fontsize=8, ncol=2)
        ax_l.grid(True, alpha=0.3)

        # 右侧文字统计
        ax_r = axes[row, 1]
        ax_r.axis('off')

        # 分层明细表格
        df = pd.DataFrame(result['分层投入'])
        table_str = df.to_string(index=False)

        stats = (
            f"【交易基本信息】\n"
            f"  总投入资金：{result['总投入资金']}\n"
            f"  买入次数：  {result['买入次数']}次\n"
            f"  累计持股：  {result['累计持股']}\n\n"
            f"【成本分析】\n"
            f"  首次买入价：{result['首次买入价格']}\n"
            f"  平均成本价：{result['平均成本价']}\n"
            f"  成本降低：  {result['成本降低']}\n\n"
            f"【卖出收益】\n"
            f"  卖出价格：  {sell['卖出价格']}\n"
            f"  卖出金额：  {sell['卖出金额']}\n"
            f"  盈利金额：  {sell['盈利金额']}\n"
            f"  盈利率：    {sell['盈利率']}\n"
            f"  达到目标：  {sell['达到目标']}\n\n"
            f"【分层投入明细】\n{table_str}"
        )
        ax_r.text(0.03, 0.97, stats, transform=ax_r.transAxes,
                  fontsize=9, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='#FFF8DC', alpha=0.7))

    plt.tight_layout()
    path = f'{out_dir}/网格交易5221场景分析.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 场景分析图表已保存至: {path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 图表2：5项关键指标对比
# ═══════════════════════════════════════════════════════════════════════════════

def create_comparison_chart(simulators: List[Tuple], config: Dict, out_dir: str):
    profit_target = config.get('profit_target_ratio', 0.3)

    labels, avg_costs, target_prices, profit_amounts, shares_list, invested_list = \
        [], [], [], [], [], []

    for _, _, result in simulators:
        extra = result['买入次数'] - 1
        labels.append(f"{extra}个\n额外买点")

        avg_cost = float(result['平均成本价'].replace('¥', ''))
        avg_costs.append(avg_cost)
        target_prices.append(avg_cost * (1 + profit_target))
        profit_amounts.append(float(result['卖出信息']['盈利金额'].replace('¥', '')))
        shares_list.append(float(result['累计持股'].replace('股', '')))
        invested_list.append(float(result['总投入资金'].replace('¥', '')))

    x         = np.arange(len(labels))
    bar_width  = 0.50

    fig = plt.figure(figsize=(16, 18))
    fig.suptitle('5221网格策略 — 4场景关键指标对比',
                 fontsize=16, fontweight='bold', y=0.99)
    gs = GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, :])

    def _bar(ax, values, title, ylabel, color, fmt, y_offset, y_min=None):
        bars = ax.bar(x, values, width=bar_width, color=color,
                      edgecolor='black', alpha=0.85, zorder=2)
        ax.plot(x, values, marker='o', color='black', lw=1.5,
                markersize=5, linestyle='--', alpha=0.7, zorder=3)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.grid(axis='y', alpha=0.3, zorder=1)
        if y_min is not None:
            ax.set_ylim(bottom=y_min)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + y_offset,
                    fmt.format(val),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    _bar(ax1, avg_costs,
         '① 平均成本价变化', '成本价（元）',
         '#4C9BE8', '¥{:.4f}', 0.02, min(avg_costs) * 0.97)

    _bar(ax2, target_prices,
         '② 预期止盈价变化（成本×1.3）', '止盈价（元）',
         '#E8834C', '¥{:.4f}', 0.02, min(target_prices) * 0.97)

    _bar(ax3, profit_amounts,
         '③ 预期盈利金额变化', '盈利金额（元）',
         '#2CA02C', '¥{:.0f}', 80)

    _bar(ax4, shares_list,
         '④ 持股数量变化', '持股数量（股）',
         '#9467BD', '{:.2f}股', 30, 0)

    _bar(ax5, invested_list,
         '⑤ 累计投入资金变化', '投入资金（元）',
         '#D62728', '¥{:.0f}', 200, 0)

    total = config.get('total_amount', 100000)
    ax5.axhline(y=total, color='black', lw=1.5, linestyle=':',
                alpha=0.6, label=f"总资金上限 ¥{total:,.0f}")
    ax5.legend(fontsize=9, loc='lower right')

    path = f'{out_dir}/网格交易5221场景对比.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"✓ 场景对比图表已保存至: {path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    OUT_DIR = 'c:/Users/明/Documents/code/obisidian-notes/乘法/测算脚本'

    # ── 策略配置 ──────────────────────────────────────────────────────────────
    # 5221：首次50%，之后每跌10%分别买20%/20%/10%，合计100%
    CONFIG = {
        'total_amount':        100000,
        'initial_price':       10.00,
        'first_buy_ratio':     0.5,              # 首次买入 50%
        'grid_drop_ratio':     0.1,              # 每跌 10% 触发一次
        'grid_invest_ratios':  [0.2, 0.2, 0.1], # ★ 5221核心：三次网格分别投20%/20%/10%
        'profit_target_ratio': 0.3,              # 目标止盈 30%（相对平均成本）
        'stock_name':          '示例股票',
    }

    # ── 各场景平均成本预算（P₀=10.00，供设计价格序列参考） ────────────────────
    #
    #   场景1（0个额外买点）：
    #     投入：5万 @ 10.00 → shares=5000.00
    #     avg_cost = 10.0000，止盈价 = 13.0000
    #
    #   场景2（1个额外买点，跌10%到9.00触发20%）：
    #     投入：5万@10 + 2万@9 → shares=5000+2222.22=7222.22，invested=7万
    #     avg_cost = 70000/7222.22 = 9.6923，止盈价 = 12.6000
    #
    #   场景3（2个额外买点，跌10%/20%分别触发20%/20%）：
    #     投入：5万@10 + 2万@9 + 2万@8 → shares=9722.22，invested=9万
    #     avg_cost = 90000/9722.22 = 9.2571，止盈价 = 12.0343
    #
    #   场景4（3个额外买点，跌10%/20%/30%分别触发20%/20%/10%，全仓）：
    #     投入：5万@10 + 2万@9 + 2万@8 + 1万@7 → shares=11150.79，invested=10万
    #     avg_cost = 100000/11150.79 = 8.9686，止盈价 = 11.6591

    # 场景1：只首次买入，直接上涨至13.00（>13.0000）
    prices_s1 = [
        10.00, 10.50, 11.00, 11.50, 12.00, 12.50, 13.00
    ]

    # 场景2：跌至9.00触发第2次买入，随后上涨至12.70（>12.6000）
    prices_s2 = [
        10.00, 9.80, 9.50, 9.00,                               # 下跌至9.00
        9.50, 10.00, 10.50, 11.00, 11.50, 12.00, 12.50, 12.70 # 回升至12.70
    ]

    # 场景3：跌至9.00、8.00各触发一次，随后上涨至12.10（>12.0343）
    prices_s3 = [
        10.00, 9.80, 9.50, 9.00,               # 下跌至9.00
        8.80, 8.50, 8.00,                       # 继续下跌至8.00
        8.50, 9.00, 9.50, 10.00, 10.50,
        11.00, 11.50, 12.00, 12.10              # 回升至12.10
    ]

    # 场景4：跌至9.00/8.00/7.00各触发一次（全仓用完），随后上涨至11.70（>11.6591）
    prices_s4 = [
        10.00, 9.80, 9.50, 9.00,               # 下跌至9.00
        8.80, 8.50, 8.00,                       # 继续下跌至8.00
        7.80, 7.50, 7.00,                       # 继续下跌至7.00
        7.50, 8.00, 8.50, 9.00, 9.50,
        10.00, 10.50, 11.00, 11.50, 11.70       # 回升至11.70
    ]

    test_scenarios = [
        {'name': '场景1：0个额外买点（只首次买入50%）',       'prices': prices_s1},
        {'name': '场景2：1个额外买点（跌10%追加20%）',        'prices': prices_s2},
        {'name': '场景3：2个额外买点（跌10%/20%追加20%/20%）', 'prices': prices_s3},
        {'name': '场景4：3个额外买点（跌10%/20%/30%，全仓）', 'prices': prices_s4},
    ]

    # ── 运行分析 ───────────────────────────────────────────────────────────────
    print("\n" + "█"*80)
    print("        5221网格交易策略 — 4个场景详细分析")
    print("        买入比例：首次50% / 第2次20% / 第3次20% / 第4次10%")
    print("█"*80)

    simulators = scenario_analysis(CONFIG, test_scenarios)

    print("\n正在生成详细走势图...")
    create_scenario_visualization(simulators, CONFIG, OUT_DIR)

    print("正在生成关键指标对比图...")
    create_comparison_chart(simulators, CONFIG, OUT_DIR)

    # ── 汇总对比表 ─────────────────────────────────────────────────────────────
    print("\n" + "█"*80)
    print("        5221策略 — 4场景汇总对比表")
    print("█"*80)

    rows = []
    for name, sim, result in simulators:
        avg_cost = float(result['平均成本价'].replace('¥', ''))
        rows.append({
            '场景':        name,
            '额外买点数':  result['买入次数'] - 1,
            '首次买入价':  result['首次买入价格'],
            '平均成本价':  result['平均成本价'],
            '目标止盈价':  f"¥{avg_cost * 1.3:.4f}",
            '成本降低':    result['成本降低'],
            '盈利金额':    result['卖出信息']['盈利金额'],
            '盈利率':      result['卖出信息']['盈利率'],
        })

    print("\n" + pd.DataFrame(rows).to_string(index=False))
    print("\n" + "█"*80)
    print("\n✓ 5221策略分析完成！")
    print(f"  输出目录：{OUT_DIR}")
    print("  1. 网格交易5221场景分析.png — 4个场景详细走势图")
    print("  2. 网格交易5221场景对比.png — 5项关键指标对比图")