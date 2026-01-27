#!/usr/bin/env python
"""
解释F1分数与其他指标的关系
"""

import numpy as np


def calculate_f1(precision, recall):
    """计算F1分数"""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def explain_metrics(accuracy, precision, recall, f1, name=""):
    """解释指标之间的关系"""
    print("=" * 60)
    if name:
        print(f"模型: {name}")
    print("=" * 60)
    print(f"准确率 (Accuracy):  {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall):    {recall:.4f}")
    print(f"F1 分数:            {f1:.4f}")
    print()
    
    # 验证F1计算
    calculated_f1 = calculate_f1(precision, recall)
    print(f"F1 计算公式: 2 * (精确率 * 召回率) / (精确率 + 召回率)")
    print(f"F1 = 2 * ({precision:.4f} * {recall:.4f}) / ({precision:.4f} + {recall:.4f})")
    print(f"F1 = {calculated_f1:.4f}")
    print()
    
    # 解释F1的特点
    print("F1 分数的特点:")
    print("1. F1 是精确率和召回率的调和平均数")
    print("2. F1 总是介于精确率和召回率之间")
    print(f"   {min(precision, recall):.4f} ≤ F1 ({f1:.4f}) ≤ {max(precision, recall):.4f}")
    print()
    
    # 解释为什么F1可能看起来"不高"
    print("为什么F1可能看起来'不高'？")
    print("- F1 是调和平均数，不是算术平均数")
    print("- 当精确率和召回率差异较大时，F1会更接近较小的那个值")
    
    if abs(precision - recall) > 0.05:
        print(f"- 你的情况：精确率({precision:.4f})和召回率({recall:.4f})差异较大")
        print(f"  差异: {abs(precision - recall):.4f}")
        print(f"  因此F1({f1:.4f})更接近较小的值({min(precision, recall):.4f})")
    
    # 算术平均数对比
    arithmetic_mean = (precision + recall) / 2
    print(f"\n对比:")
    print(f"  算术平均数: (精确率 + 召回率) / 2 = {arithmetic_mean:.4f}")
    print(f"  调和平均数 (F1): {f1:.4f}")
    print(f"  差异: {arithmetic_mean - f1:.4f}")
    print()
    
    # 解释准确率
    print("准确率 vs F1:")
    print("- 准确率考虑所有样本（包括负类）")
    print("- F1只考虑正类的预测质量")
    print("- 当类别不平衡时，准确率可能被多数类主导")
    print()
    
    # 判断类别是否平衡
    if abs(accuracy - f1) > 0.05:
        print(f"⚠️  注意：准确率({accuracy:.4f})和F1({f1:.4f})差异较大")
        print("   这可能表明数据集存在类别不平衡问题")
    print()


def main():
    # 第一组数据
    explain_metrics(
        accuracy=0.8099,
        precision=0.8067,
        recall=0.8953,
        f1=0.8487,
        name="模型1"
    )
    
    # 第二组数据
    explain_metrics(
        accuracy=0.8640,
        precision=0.8533,
        recall=0.9318,
        f1=0.8909,
        name="模型2"
    )
    
    # 对比分析
    print("=" * 60)
    print("对比分析")
    print("=" * 60)
    print("模型2相比模型1的改进:")
    print(f"  准确率: 0.8099 → 0.8640 (+{0.8640-0.8099:.4f})")
    print(f"  精确率: 0.8067 → 0.8533 (+{0.8533-0.8067:.4f})")
    print(f"  召回率: 0.8953 → 0.9318 (+{0.9318-0.8953:.4f})")
    print(f"  F1:     0.8487 → 0.8909 (+{0.8909-0.8487:.4f})")
    print()
    
    print("模型2的F1 (0.8909) 实际上比模型1的F1 (0.8487) 更高！")
    print()
    print("如果感觉F1'不高'，可能是因为:")
    print("1. F1是调和平均数，总是小于算术平均数")
    print("2. 当精确率和召回率差异大时，F1会更接近较小的值")
    print("3. 模型2的精确率(0.8533)和召回率(0.9318)差异较大(0.0785)")
    print("   所以F1(0.8909)更接近精确率，而不是召回率")
    print()
    print("实际上，F1 = 0.8909 是一个很好的分数！")


if __name__ == '__main__':
    main()
