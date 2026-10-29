import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体，防止图表中文乱码
# Windows 用 SimHei，Mac 用 Arial Unicode MS
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取数据 (注意编码用 utf-8-sig 解决带 BOM 的 CSV 问题)
df_price = pd.read_csv('data/各省份住宅商品房平均销售价格.csv', encoding='utf-8-sig')
df_income = pd.read_csv('data/各省份城镇居民人均可支配收入.csv', encoding='utf-8-sig')

# 2. 清洗列名 (去除可能存在的 \t 和空格)
df_price.columns = df_price.columns.str.strip().str.replace('\t', '')
df_income.columns = df_income.columns.str.strip().str.replace('\t', '')

# 3. 清洗地区名 (去掉“省”、“市”、“自治区”，方便后续匹配)
def clean_region_name(region):
    return region.strip().replace('省', '').replace('市', '')\
                 .replace('壮族自治区', '').replace('回族自治区', '')\
                 .replace('维吾尔自治区', '').replace('自治区', '')

df_price['地区'] = df_price['地区'].apply(clean_region_name)
df_income['地区'] = df_income['地区'].apply(clean_region_name)

# 4. 宽表转长表 (只保留 2015-2023 年的数据)
years_to_keep = ['地区', '2015年', '2016年', '2017年', '2018年',
                 '2019年', '2020年', '2021年', '2022年', '2023年']

price_long = df_price[years_to_keep].melt(id_vars=['地区'], var_name='年份', value_name='住宅商品房价平均销售价格')
income_long = df_income[years_to_keep].melt(id_vars=['地区'], var_name='年份', value_name='城镇居民人均可支配收入')

# 提取年份数字
price_long['年份'] = price_long['年份'].str.extract(r'(\d{4})').astype(int)
income_long['年份'] = income_long['年份'].str.extract(r'(\d{4})').astype(int)

print("清洗后的房价数据前3行：")
print(price_long.head(20))

# 1. 多表合并
df = pd.merge(price_long, income_long, on=['地区', '年份'], how='inner')

# 2. 映射四大经济区域
region_map = {
    '北京': '东部', '天津': '东部', '河北': '东部', '上海': '东部',
    '江苏': '东部', '浙江': '东部', '福建': '东部', '山东': '东部',
    '广东': '东部', '海南': '东部',

    '山西': '中部', '安徽': '中部', '江西': '中部', '河南': '中部',
    '湖北': '中部', '湖南': '中部',

    '内蒙古': '西部', '广西': '西部', '重庆': '西部', '四川': '西部',
    '贵州': '西部', '云南': '西部', '西藏': '西部', '陕西': '西部',
    '甘肃': '西部', '青海': '西部', '宁夏': '西部', '新疆': '西部',

    '辽宁': '东北', '吉林': '东北', '黑龙江': '东北'
}
df['经济区域'] = df['地区'].map(region_map)
df = df.dropna(subset=['经济区域']) # 防止有遗漏

# 3. 计算核心指标：房价收入比 (90平米/户，3人/户)
# ⚠️ 注意这里使用的是你的长列名
df['房价收入比'] = (df['住宅商品房价平均销售价格'] * 90) / (df['城镇居民人均可支配收入'] * 3)

# 4. 计算复合年均增长率 (CAGR 2015-2023，跨度8年)
def calc_cagr(group):
    start = group[group['年份'] == 2015]
    end = group[group['年份'] == 2023]
    if len(start) == 0 or len(end) == 0:
        return pd.Series({'房价CAGR': np.nan, '收入CAGR': np.nan})

    years = 8
    price_cagr = (end['住宅商品房价平均销售价格'].values[0] / start['住宅商品房价平均销售价格'].values[0]) ** (1/years) - 1
    income_cagr = (end['城镇居民人均可支配收入'].values[0] / start['城镇居民人均可支配收入'].values[0]) ** (1/years) - 1
    return pd.Series({'房价CAGR': price_cagr, '收入CAGR': income_cagr})

cagr_df = df.groupby('地区').apply(calc_cagr).reset_index()
cagr_df['增速差'] = cagr_df['房价CAGR'] - cagr_df['收入CAGR']

# 把 CAGR 合并回主表
df = pd.merge(df, cagr_df, on='地区', how='left')

# 5. 计算区域不平衡度 (变异系数 = 标准差 / 均值)
cv_by_year = df.groupby('年份')['房价收入比'].agg(['mean', 'std'])
cv_by_year['变异系数'] = cv_by_year['std'] / cv_by_year['mean']

print("✅ 指标计算完成！前3行数据如下：")
print(df[['地区', '年份', '住宅商品房价平均销售价格', '城镇居民人均可支配收入', '房价收入比']].head(3))

# 确保存储图片的文件夹存在
os.makedirs('reports/figures', exist_ok=True)

# ================= 图1：2023年四大区域平均房价收入比 =================
d23 = df[df['年份'] == 2023]
region_mean = d23.groupby('经济区域')['房价收入比'].mean().sort_values(ascending=False)

plt.figure(figsize=(8, 5))
bars = plt.bar(region_mean.index, region_mean.values, color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'])
plt.axhline(y=15, color='gray', linestyle='--', label='国际警戒线 (15)')
plt.title('2023年四大经济区域平均房价收入比', fontsize=14)
plt.ylabel('房价收入比', fontsize=12)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.2, round(yval, 2), ha='center')

plt.legend()
plt.tight_layout()
plt.savefig('reports/figures/fig1_region_compare.png', dpi=200)
plt.show()


# ================= 图2：区域不平衡度趋势 =================
plt.figure(figsize=(8, 5))
plt.plot(cv_by_year.index, cv_by_year['变异系数'], marker='o', color='purple', linewidth=2)
plt.title('2015-2023 省级房价收入比变异系数 (区域分化程度)', fontsize=14)
plt.xlabel('年份', fontsize=12)
plt.ylabel('变异系数 (越大代表区域差异越大)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('reports/figures/fig2_cv_trend.png', dpi=200)
plt.show()


# ================= 图3：增速散点图 (找剪刀差省份) =================
plt.figure(figsize=(9, 6))
plt.scatter(cagr_df['收入CAGR']*100, cagr_df['房价CAGR']*100, alpha=0.7, color='teal')

max_val = max(cagr_df['收入CAGR'].max(), cagr_df['房价CAGR'].max()) * 100 + 1
min_val = min(cagr_df['收入CAGR'].min(), cagr_df['房价CAGR'].min()) * 100 - 1
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='增速相等线 (45度)')

top_risky = cagr_df.sort_values(by='增速差', ascending=False).head(3)
for _, row in top_risky.iterrows():
    plt.annotate(row['地区'],
                 (row['收入CAGR']*100, row['房价CAGR']*100),
                 xytext=(5, 5), textcoords='offset points', color='red', weight='bold')

plt.title('2015-2023 各省住宅均价与收入复合增速 (红线以上为风险区)', fontsize=14)
plt.xlabel('城镇居民人均可支配收入复合增速 (%)', fontsize=12)
plt.ylabel('住宅商品房平均销售价格复合增速 (%)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('reports/figures/fig3_scatter.png', dpi=200)
plt.show()

# 确保 processed 文件夹存在
os.makedirs('data/processed', exist_ok=True)

# 导出最终表格
df.to_csv('data/processed/clean_housing_data.csv', index=False, encoding='utf-8-sig')
print("✅ 数据已成功保存至 'data/processed/clean_housing_data.csv'\n")

# 打印核心业务发现
print("📊 核心数据洞察（直接抄进你的报告里）：")
print("-" * 50)
print(f"1. 2023年，【{region_mean.index[0]}】地区的平均房价收入比最高，达到 {region_mean.values[0]:.2f}，是【{region_mean.index[-1]}】地区的 {region_mean.values[0]/region_mean.values[-1]:.2f} 倍。")
print(f"2. 区域不平衡度（变异系数）从 2015 年的 {cv_by_year.loc[2015, '变异系数']:.3f}，变化至 2023 年的 {cv_by_year.loc[2023, '变异系数']:.3f}。")
print("3. 剪刀差风险最高（房价涨得远比收入快）的 3 个省份是：")
for _, row in top_risky.iterrows():
    print(f"   👉 {row['地区']}: 房价年均增速 {row['房价CAGR']*100:.2f}%  vs  收入年均增速 {row['收入CAGR']*100:.2f}%")
print("-" * 50)