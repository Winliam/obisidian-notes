"""
股价网格交易策略推演脚本
根据定额资金进行网格交易，逐步摊低成本价，达到目标收益率后卖出
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np

# 设置标准输出编码为UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# 设置matplotlib为非交互模式
import matplotlib
matplotlib.use('Agg')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'DejaVu Sans']
plt.rcParams['font.serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10


def verify_font_config():
    import matplotlib.font_manager as fm
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi']
    found_fonts = [f for f in chinese_fonts if f in available_fonts]
    if found_fonts:
        selected_font = found_fonts[0]
        plt.rcParams['font.sans-serif'].remove(selected_font)
        plt.rcParams['font.sans-serif'].insert(0, selected_font)
        print(f"✓ 检测到中文字体: {selected_font}")
    else:
        print("⚠ 警告: 未检测到系统中文字体，将使用默认字体")
    return plt.rcParams['font.sans-serif'][0]


verify_font_config()


class GridTradingSimulator:
    """网格交易策略模拟器"""

    def __init__(self, config: Dict):
        self.total_amount = config.get('total_amount', 100000)
        self.initial_price = config.get('initial_price', 10.0)
        self.first_buy_ratio = config.get('first_buy_ratio', 0.5)
        self.grid_drop_ratio = config.get('grid_drop_ratio', 0.1)
        self.grid_invest_ratio = config.get('grid_invest_ratio', 0.1)
        self.profit_target_ratio = config.get('profit_target_ratio', 0.3)
        self.stock_name = config.get('stock_name', '某股票')
        self._validate_config()

        self.buy_records = []
        self.total_invested = 0
        self.total_shares = 0
        self.avg_cost = 0
        self.price_history = []
        self.cost_history = []
        self.target_price_history = []
        self.loss_at_buy = []

    def _validate_config(self):
        if self.total_amount <= 0:
            raise ValueError("总投入资金必须大于0")
        if self.initial_price <= 0:
            raise ValueError("初始价格必须大于0")
        if not (0 < self.first_buy_ratio <= 1):
            raise ValueError("首次买入比例必须在0到1之间")
        if not (0 < self.grid_drop_ratio <= 1):
            raise ValueError("网格下跌比例必须在0到1之间")
        if not (0 < self.grid_invest_ratio <= 1):
            raise ValueError("网格投入比例必须在0到1之间")
        if not (0 < self.profit_target_ratio <= 2):
            raise ValueError("目标盈利比例必须在0到2之间")

    def simulate_trading(self, price_list: List[float]) -> Dict:
        # ── 重置状态 ──────────────────────────────────────────────
        self.buy_records = []
        self.total_invested = 0
        self.total_shares = 0
        self.avg_cost = 0
        self.price_history = []
        self.cost_history = []
        self.target_price_history = []
        self.loss_at_buy = []

        sell_price = None
        sell_date = None

        for date_idx, current_price in enumerate(price_list):
            self.price_history.append(current_price)

            # ── 步骤1：执行买入（首次 or 网格触发） ──────────────
            if date_idx == 0:
                # 首次买入
                self._buy(current_price, date_idx, is_first=True)

            else:
                # 检查是否触发网格买入
                first_buy_price = self.buy_records[0]['price']
                price_drop_ratio = (first_buy_price - current_price) / first_buy_price

                if price_drop_ratio > 0:
                    # +1e-9 修正浮点截断误差：如 0.3/0.1 = 2.9999...，int() 会错误截断为 2
                    grid_level = int(price_drop_ratio / self.grid_drop_ratio + 1e-9)
                    current_buy_count = len(self.buy_records)
                    remaining = self.total_amount - self.total_invested

                    # 进入了新的网格层级 且 还有剩余资金 → 执行买入
                    if grid_level > (current_buy_count - 1) and remaining > 0:
                        # 记录买入前的浮动亏损
                        current_loss = (self.avg_cost - current_price) * self.total_shares
                        self.loss_at_buy.append(current_loss)
                        self._buy(current_price, date_idx)

            # ── 步骤2：记录历史（买入执行完毕之后再记录） ────────
            # 修复：原代码在执行买入之前就记录，导致买入当天的成本值错误（如首日记录为0）
            self.cost_history.append(self.avg_cost)
            self.target_price_history.append(
                self.avg_cost * (1 + self.profit_target_ratio) if self.avg_cost > 0 else 0
            )

            # ── 步骤3：检查止盈条件（首次买入当天跳过，成本=价格盈利为0） ──
            if date_idx > 0 and self.total_shares > 0:
                if self._check_sell_condition(current_price):
                    # 卖出价精确取 avg_cost × (1 + profit_target_ratio)，
                    # 而非序列中的离散价格（如 13.00），确保各场景止盈数值一致可比
                    sell_price = self.avg_cost * (1 + self.profit_target_ratio)
                    sell_date = date_idx
                    break

        return self._generate_report(sell_price, sell_date)

    def _buy(self, price: float, date_idx: int, is_first: bool = False):
        if is_first:
            amount = self.total_amount * self.first_buy_ratio
        else:
            amount = self.total_amount * self.grid_invest_ratio

        # 不超过剩余额度
        remaining = self.total_amount - self.total_invested
        if amount > remaining:
            amount = remaining

        shares = amount / price
        self.total_invested += amount
        self.total_shares += shares
        self.avg_cost = self.total_invested / self.total_shares

        self.buy_records.append({
            'date_idx': date_idx,
            'price': price,
            'amount': amount,
            'shares': shares,
            'total_invested': self.total_invested,
            'total_shares': self.total_shares,
            'avg_cost': self.avg_cost,
            'grid_level': len(self.buy_records)  # 记录前的长度即为当前层级编号
        })

    def _check_sell_condition(self, current_price: float) -> bool:
        if self.total_shares == 0:
            return False
        profit_ratio = (current_price - self.avg_cost) / self.avg_cost
        return profit_ratio >= self.profit_target_ratio

    def _generate_report(self, sell_price: float = None, sell_date: int = None) -> Dict:
        if sell_price is not None:
            sell_amount = self.total_shares * sell_price
            profit_amount = sell_amount - self.total_invested
            profit_ratio = profit_amount / self.total_invested
            reach_target = True
        else:
            sell_amount = 0
            profit_amount = 0
            profit_ratio = 0
            reach_target = False

        if len(self.buy_records) > 1:
            first_buy_price = self.buy_records[0]['price']
            cost_reduction = (first_buy_price - self.avg_cost) / first_buy_price
            cost_reduction_amount = first_buy_price - self.avg_cost
        else:
            cost_reduction = 0
            cost_reduction_amount = 0

        layers_info = []
        for record in self.buy_records:
            layers_info.append({
                '网格层级': record['grid_level'],
                '买入价格': f"{record['price']:.4f}",
                '购买股数': f"{record['shares']:.4f}",
                '投入金额': f"{record['amount']:.2f}",
                '累计投入': f"{record['total_invested']:.2f}",
                '平均成本': f"{record['avg_cost']:.4f}"
            })

        return {
            '股票': self.stock_name,
            '总投入资金': f"¥{self.total_invested:.2f}",
            '买入次数': len(self.buy_records),
            '首次买入价格': f"¥{self.buy_records[0]['price']:.4f}",
            '平均成本价': f"¥{self.avg_cost:.4f}",
            '成本降低': f"{cost_reduction:.2%} (¥{cost_reduction_amount:.4f})",
            '累计持股': f"{self.total_shares:.4f}股",
            '分层投入': layers_info,
            '卖出信息': {
                '卖出价格': f"¥{sell_price:.4f}" if sell_price is not None else "未卖出",
                '卖出日期': sell_date if sell_date is not None else "N/A",
                '卖出金额': f"¥{sell_amount:.2f}",
                '盈利金额': f"¥{profit_amount:.2f}",
                '盈利率': f"{profit_ratio:.2%}",
                '达到目标': "是" if reach_target else "否"
            }
        }


def print_report(result: Dict):
    print("\n" + "="*80)
    print(f"   股票网格交易策略分析报告 - {result['股票']}")
    print("="*80)
    print(f"\n【基本信息】")
    print(f"  总投入资金:   {result['总投入资金']}")
    print(f"  买入次数:     {result['买入次数']}次")
    print(f"  累计持股:     {result['累计持股']}")
    print(f"\n【成本分析】")
    print(f"  首次买入价格: {result['首次买入价格']}")
    print(f"  平均成本价:  {result['平均成本价']}")
    print(f"  成本降低:    {result['成本降低']}")
    print(f"\n【分层投入明细】")
    df_layers = pd.DataFrame(result['分层投入'])
    print(df_layers.to_string(index=False))
    print(f"\n【卖出信息】")
    sell_info = result['卖出信息']
    print(f"  卖出价格:     {sell_info['卖出价格']}")
    print(f"  卖出金额:     {sell_info['卖出金额']}")
    print(f"  盈利金额:     {sell_info['盈利金额']}")
    print(f"  盈利率:       {sell_info['盈利率']}")
    print(f"  达到目标:     {sell_info['达到目标']}")
    print("\n" + "="*80)


def scenario_analysis(config: Dict, test_scenarios: List[Dict]):
    print("\n\n" + "="*80)
    print("   场景分析：不同价格走势下的收益情况")
    print("="*80)

    results_summary = []
    simulators = []

    for scenario in test_scenarios:
        simulator = GridTradingSimulator(config)
        result = simulator.simulate_trading(scenario['prices'])
        simulators.append((scenario['name'], simulator, result))

        sell_info = result['卖出信息']
        results_summary.append({
            '场景': scenario['name'],
            '最低价': f"¥{min(scenario['prices']):.4f}",
            '最高价': f"¥{max(scenario['prices']):.4f}",
            '平均成本': result['平均成本价'],
            '卖出价格': sell_info['卖出价格'],
            '盈利率': sell_info['盈利率'],
            '达到目标': sell_info['达到目标']
        })

    df_summary = pd.DataFrame(results_summary)
    print("\n" + df_summary.to_string(index=False))
    print("\n" + "="*80)
    return simulators


def create_scenario_visualization(simulators: List[Tuple], config: Dict):
    num_scenarios = len(simulators)
    fig, axes = plt.subplots(num_scenarios, 2, figsize=(16, 5 * num_scenarios))
    if num_scenarios == 1:
        axes = axes.reshape(1, -1)

    color_price = '#1f77b4'
    color_cost = '#ff7f0e'
    color_target = '#2ca02c'

    for row, (scenario_name, simulator, result) in enumerate(simulators):
        ax_left = axes[row, 0]
        price_history = simulator.price_history
        cost_history = simulator.cost_history
        target_history = simulator.target_price_history
        x_axis = range(len(price_history))

        ax_left.plot(x_axis, price_history, marker='o', linewidth=2, label='股价',
                     color=color_price, markersize=6)
        ax_left.plot(x_axis, cost_history, marker='s', linewidth=2, label='平均成本价',
                     color=color_cost, markersize=5)
        ax_left.plot(x_axis, target_history, marker='^', linewidth=2, linestyle='--',
                     label='目标止盈价', color=color_target, markersize=5)

        for buy_record in simulator.buy_records:
            ax_left.scatter(buy_record['date_idx'], buy_record['price'],
                            s=200, marker='v', color='red', zorder=5,
                            edgecolors='darkred', linewidths=2)

        sell_info = result['卖出信息']
        if sell_info['卖出价格'] != '未卖出':
            sell_price = float(sell_info['卖出价格'].replace('¥', ''))
            sell_date = sell_info['卖出日期']
            if sell_date != 'N/A':
                ax_left.scatter(sell_date, sell_price, s=200, marker='^', color='green',
                                zorder=5, edgecolors='darkgreen', linewidths=2)

        ax_left.set_title(f'场景 {row + 1}: {scenario_name}\n价格走势与成本分析',
                          fontsize=12, fontweight='bold', pad=10)
        ax_left.set_xlabel('时间周期', fontsize=10)
        ax_left.set_ylabel('价格（元）', fontsize=10)
        ax_left.legend(loc='best', fontsize=9)
        ax_left.grid(True, alpha=0.3)

        ax_right = axes[row, 1]
        ax_right.axis('off')
        stats_text = f"""
【交易基本信息】
总投入资金：{result['总投入资金']}
买入次数：{result['买入次数']}次
累计持股：{result['累计持股']}

【成本分析】
首次买入价格：{result['首次买入价格']}
平均成本价：{result['平均成本价']}
成本降低：{result['成本降低']}

【卖出收益】
卖出价格：{result['卖出信息']['卖出价格']}
卖出金额：{result['卖出信息']['卖出金额']}
盈利金额：{result['卖出信息']['盈利金额']}
盈利率：{result['卖出信息']['盈利率']}
达到目标：{result['卖出信息']['达到目标']}
"""
        ax_right.text(0.05, 0.95, stats_text, transform=ax_right.transAxes,
                      fontsize=10, verticalalignment='top', fontfamily='sans-serif',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    out_path = 'c:/Users/明/Documents/code/obisidian-notes/乘法/测算脚本/网格交易场景分析.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 场景分析图表已保存至: 乘法/测算脚本/网格交易场景分析.png")
    plt.close()


def create_comparison_chart(simulators: List[Tuple], config: Dict):
    """
    多场景对比图表，展示5项关键指标随网格买入次数增加的变化趋势：
      1. 平均成本价变化
      2. 预期止盈价变化（avg_cost × 1.3）
      3. 预期盈利金额变化
      4. 持股数量变化
      5. 累计投入资金变化
    """
    profit_target = config.get('profit_target_ratio', 0.3)

    labels = []
    avg_costs       = []   # 平均成本价
    target_prices   = []   # 预期止盈价 = avg_cost × (1 + profit_target)
    profit_amounts  = []   # 预期盈利金额
    total_shares    = []   # 持股数量
    total_invested  = []   # 累计投入资金

    for scenario_name, simulator, result in simulators:
        extra_buys = result['买入次数'] - 1
        labels.append(f"{extra_buys}个\n额外买点")

        avg_cost = float(result['平均成本价'].replace('¥', ''))
        avg_costs.append(avg_cost)
        target_prices.append(avg_cost * (1 + profit_target))
        profit_amounts.append(float(result['卖出信息']['盈利金额'].replace('¥', '')))
        total_shares.append(float(result['累计持股'].replace('股', '')))
        total_invested.append(float(result['总投入资金'].replace('¥', '')))

    x = np.arange(len(labels))
    bar_width = 0.55

    # ── 整体布局：3行2列，第5格居中跨列 ──────────────────────────
    fig = plt.figure(figsize=(16, 18))
    fig.suptitle('网格交易策略 — 6场景关键指标对比', fontsize=16, fontweight='bold', y=0.98)

    # 用 GridSpec 让第5个图居中
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, :])   # 第5图占满第3行

    def _bar_chart(ax, values, title, ylabel, color, fmt, y_offset, y_min=None):
        """通用柱状图绘制，叠加折线趋势"""
        bars = ax.bar(x, values, width=bar_width, color=color,
                      edgecolor='black', alpha=0.85, zorder=2)
        ax.plot(x, values, marker='o', color='black', linewidth=1.5,
                markersize=5, zorder=3, linestyle='--', alpha=0.7)
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
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    # ① 平均成本价
    _bar_chart(ax1, avg_costs,
               title='① 平均成本价变化',
               ylabel='成本价（元）',
               color='#4C9BE8',
               fmt='¥{:.4f}',
               y_offset=0.05,
               y_min=min(avg_costs) * 0.97)

    # ② 预期止盈价
    _bar_chart(ax2, target_prices,
               title='② 预期止盈价变化（成本×1.3）',
               ylabel='止盈价（元）',
               color='#E8834C',
               fmt='¥{:.4f}',
               y_offset=0.05,
               y_min=min(target_prices) * 0.97)

    # ③ 预期盈利金额
    _bar_chart(ax3, profit_amounts,
               title='③ 预期盈利金额变化',
               ylabel='盈利金额（元）',
               color='#2CA02C',
               fmt='¥{:.0f}',
               y_offset=100)

    # ④ 持股数量
    _bar_chart(ax4, total_shares,
               title='④ 持股数量变化',
               ylabel='持股数量（股）',
               color='#9467BD',
               fmt='{:.2f}股',
               y_offset=50,
               y_min=0)

    # ⑤ 累计投入资金（跨列居中）
    _bar_chart(ax5, total_invested,
               title='⑤ 累计投入资金变化',
               ylabel='投入资金（元）',
               color='#D62728',
               fmt='¥{:.0f}',
               y_offset=300,
               y_min=0)
    # 在⑤上补充一条"总资金上限"参考线
    ax5.axhline(y=config.get('total_amount', 100000), color='black',
                linewidth=1.5, linestyle=':', alpha=0.6, label=f"总资金上限 ¥{config.get('total_amount',100000):,.0f}")
    ax5.legend(fontsize=9, loc='lower right')

    out_path = 'c:/Users/明/Documents/code/obisidian-notes/乘法/测算脚本/网格交易场景对比.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✓ 场景对比图表已保存至: 乘法/测算脚本/网格交易场景对比.png")
    plt.close()


if __name__ == "__main__":
    # ========== 配置参数 ==========
    CONFIG = {
        'total_amount': 100000,
        'initial_price': 10.00,
        'first_buy_ratio': 0.5,       # 首次买入 50%
        'grid_drop_ratio': 0.1,       # 每跌 10% 触发一次网格
        'grid_invest_ratio': 0.1,     # 每次网格投入 10%
        'profit_target_ratio': 0.3,   # 目标止盈 30%（相对平均成本）
        'stock_name': '示例股票'
    }

    # ========== 6个场景：0-5个额外网格买入点 ==========
    # 价格路径设计原则：
    #   - 下跌段：每跌 10% 触发一次网格买入
    #   - 上涨段：最终价格必须 >= 平均成本 × 1.3，确保达到止盈
    #
    # 各场景实际平均成本（含首次买入50%资金）：
    #   0次额外买入：avg=10.000，止盈=13.000
    #   1次额外买入：avg≈9.818，止盈≈12.764
    #   2次额外买入：avg≈9.474，止盈≈12.316
    #   3次额外买入：avg≈9.130，止盈≈11.869
    #   4次额外买入：avg≈8.786，止盈≈11.422
    #   5次额外买入：avg≈8.442，止盈≈10.975

    # 场景1：0个额外买入点（只首次买入）
    scenario_0_buys = [
        10.00, 10.50, 11.00, 11.50, 12.00, 12.50, 13.00
    ]

    # 场景2：1个额外买入点（跌到9.00触发，止盈≈12.764，上涨至13.00）
    scenario_1_buy = [
        10.00, 10.00, 9.80, 9.70, 9.50, 9.00,
        9.50, 10.00, 10.50, 11.00, 11.50, 12.00, 12.50, 13.00
    ]

    # 场景3：2个额外买入点（跌到9.00和8.00各触发一次，止盈≈12.316，上涨至12.50）
    scenario_2_buys = [
        10.00, 10.00, 9.80, 9.50, 9.00,
        9.00, 8.80, 8.50, 8.00,
        8.50, 9.00, 9.50, 10.00, 10.50, 11.00, 11.50, 12.00, 12.50
    ]

    # 场景4：3个额外买入点（跌到9/8/7各触发，止盈≈11.869，上涨至12.00）
    scenario_3_buys = [
        10.00, 10.00, 9.80, 9.50, 9.00,
        9.00, 8.80, 8.50, 8.00,
        8.00, 7.80, 7.50, 7.00,
        7.50, 8.00, 8.50, 9.00, 9.50, 10.00, 10.50, 11.00, 11.50, 12.00
    ]

    # 场景5：4个额外买入点（跌到9/8/7/6各触发，止盈≈11.422，上涨至12.00）
    scenario_4_buys = [
        10.00, 10.00, 9.80, 9.50, 9.00,
        9.00, 8.80, 8.50, 8.00,
        8.00, 7.80, 7.50, 7.00,
        7.00, 6.80, 6.50, 6.00,
        6.50, 7.00, 7.50, 8.00, 8.50, 9.00, 9.50, 10.00, 10.50, 11.00, 11.50, 12.00
    ]

    # 场景6：5个额外买入点（跌到9/8/7/6/5各触发，止盈≈10.975，上涨至11.50）
    scenario_5_buys = [
        10.00, 10.00, 9.80, 9.50, 9.00,
        9.00, 8.80, 8.50, 8.00,
        8.00, 7.80, 7.50, 7.00,
        7.00, 6.80, 6.50, 6.00,
        6.00, 5.80, 5.50, 5.00,
        5.50, 6.00, 6.50, 7.00, 7.50, 8.00, 8.50, 9.00, 9.50, 10.00, 10.50, 11.00, 11.50
    ]

    test_scenarios = [
        {'name': '场景1：0个购买点（只首次买入）', 'prices': scenario_0_buys},
        {'name': '场景2：1个购买点（跌10%）',     'prices': scenario_1_buy},
        {'name': '场景3：2个购买点（跌20%）',     'prices': scenario_2_buys},
        {'name': '场景4：3个购买点（跌30%）',     'prices': scenario_3_buys},
        {'name': '场景5：4个购买点（跌40%）',     'prices': scenario_4_buys},
        {'name': '场景6：5个购买点（跌50%）',     'prices': scenario_5_buys},
    ]

    print("\n" + "█"*80)
    print("          股价买入卖出网格策略 - 6个场景详细分析")
    print("█"*80)

    simulators = scenario_analysis(CONFIG, test_scenarios)

    print("\n正在生成详细的图表分析...")
    create_scenario_visualization(simulators, CONFIG)

    print("正在生成对比分析图表...")
    create_comparison_chart(simulators, CONFIG)

    print("\n" + "█"*80)
    print("          详细的交易明细对比表")
    print("█"*80)

    comparison_data = []
    for scenario_name, simulator, result in simulators:
        first_price = float(result['首次买入价格'].replace('¥', ''))
        avg_cost = float(result['平均成本价'].replace('¥', ''))
        comparison_data.append({
            '场景': scenario_name,
            '额外购买点数': result['买入次数'] - 1,
            '首次价格': f"¥{first_price:.2f}",
            '平均成本': f"¥{avg_cost:.4f}",
            '成本降低': result['成本降低'],
            '盈利金额': result['卖出信息']['盈利金额'],
            '盈利率': result['卖出信息']['盈利率'],
        })

    df_comparison = pd.DataFrame(comparison_data)
    print("\n" + df_comparison.to_string(index=False))
    print("\n" + "█"*80)

    print("\n✓ 分析完成！")
    print("  已生成以下文件：")
    print("  1. 网格交易场景分析.png - 每个场景的详细走势图")
    print("  2. 网格交易场景对比.png - 多场景对比分析图")
    print("\n  参数修改建议：")
    print("    - 修改 CONFIG 字典中的参数来自定义策略")
    print("    - 修改各 scenario_*_buys 中的价格序列来自定义走势")