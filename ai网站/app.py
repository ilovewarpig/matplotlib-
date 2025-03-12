import matplotlib
matplotlib.use('Agg')
from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import io
import base64
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

# 设置Matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

app = Flask(__name__)

# 预设配色方案
COLOR_SCHEMES = {
    'viridis': 'viridis',
    'plasma': 'plasma',
    'inferno': 'inferno',
    'magma': 'magma',
    'cividis': 'cividis',
    'coolwarm': 'coolwarm',
    'rainbow': 'rainbow',
    'RdYlBu': 'RdYlBu',
    'spectral': 'nipy_spectral'
}

def generate_plot(plot_func, color_scheme='viridis'):
    plt.figure(figsize=(10, 6))
    img = io.BytesIO()
    plot_func(color_scheme)
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

def plot_line_chart(color_scheme):
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    plt.plot(x, y, color=plt.cm.get_cmap(COLOR_SCHEMES[color_scheme])(0.6), linewidth=2)
    plt.title('优雅的正弦曲线', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.fill_between(x, y, alpha=0.2, color=plt.cm.get_cmap(COLOR_SCHEMES[color_scheme])(0.3))

def plot_bar_chart(color_scheme):
    categories = ['A', 'B', 'C', 'D', 'E']
    values = np.random.randint(10, 50, size=5)
    colors = [plt.cm.get_cmap(COLOR_SCHEMES[color_scheme])(i/4) for i in range(5)]
    plt.bar(categories, values, color=colors)
    plt.title('多彩柱状图', fontsize=14)
    for i, v in enumerate(values):
        plt.text(i, v + 1, str(v), ha='center')

def plot_scatter(color_scheme):
    np.random.seed(42)
    x = np.random.normal(0, 1, 100)
    y = x + np.random.normal(0, 0.4, 100)
    plt.scatter(x, y, c=y, cmap=COLOR_SCHEMES[color_scheme], alpha=0.6)
    plt.colorbar(label='值')
    plt.title('渐变色散点图', fontsize=14)

def plot_heatmap_chart(color_scheme):
    data = np.random.rand(8, 8)
    sns.heatmap(data, cmap=COLOR_SCHEMES[color_scheme], annot=True, fmt='.2f')
    plt.title('热力图展示', fontsize=14)

def plot_pie_chart(color_scheme):
    labels = ['A', 'B', 'C', 'D']
    sizes = [15, 30, 45, 10]
    colors = [plt.cm.get_cmap(COLOR_SCHEMES[color_scheme])(i/3) for i in range(4)]
    explode = (0.1, 0, 0, 0)
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)
    plt.title('精美饼图', fontsize=14)

def plot_box_chart(color_scheme):
    np.random.seed(42)
    data = [np.random.normal(0, std, 100) for std in range(1, 4)]
    plt.boxplot(data, patch_artist=True,
                boxprops=dict(facecolor=plt.cm.get_cmap(COLOR_SCHEMES[color_scheme])(0.6), color='black'),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'),
                medianprops=dict(color='white'))
    plt.title('箱线图分布', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

def plot_heatmap(color_scheme):
    data = np.random.rand(8, 8)
    sns.heatmap(data, cmap=COLOR_SCHEMES[color_scheme], annot=True, fmt='.2f')
    plt.title('热力图展示', fontsize=14)

def plot_violin(color_scheme):
    np.random.seed(42)
    data = [np.random.normal(0, std, 100) for std in range(1, 4)]
    parts = plt.violinplot(data)
    for pc in parts['bodies']:
        pc.set_facecolor(plt.cm.get_cmap(COLOR_SCHEMES[color_scheme])(0.6))
        pc.set_alpha(0.7)
    plt.title('小提琴图', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

def plot_3d_surface(color_scheme):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))
    surf = ax.plot_surface(X, Y, Z, cmap=color_scheme)
    plt.colorbar(surf)
    plt.title('3D曲面图', fontsize=14)

def plot_radar(color_scheme):
    categories = ['A', 'B', 'C', 'D', 'E']
    values = np.random.randint(50, 100, size=5)
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
    values = np.concatenate((values, [values[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    plt.polar(angles, values, color=plt.cm.get_cmap(COLOR_SCHEMES[color_scheme])(0.6))
    plt.fill(angles, values, alpha=0.25, color=plt.cm.get_cmap(COLOR_SCHEMES[color_scheme])(0.3))
    plt.title('雷达图', fontsize=14)

def plot_bubble(color_scheme):
    np.random.seed(42)
    x = np.random.rand(20)
    y = np.random.rand(20)
    size = np.random.rand(20) * 500
    colors = np.random.rand(20)
    plt.scatter(x, y, s=size, c=colors, alpha=0.6, cmap=color_scheme)
    plt.colorbar(label='颜色值')
    plt.title('气泡图', fontsize=14)

def plot_density(color_scheme):
    np.random.seed(42)
    data = np.random.normal(0, 1, 1000)
    sns.kdeplot(data=data, fill=True, color=plt.cm.get_cmap(COLOR_SCHEMES[color_scheme])(0.6))
    plt.title('核密度图', fontsize=14)

def plot_ridgeline(color_scheme):
    np.random.seed(42)
    data = [np.random.normal(i, 1, 100) for i in range(5)]
    for i, d in enumerate(data):
        sns.kdeplot(data=d, fill=True, alpha=0.5, color=plt.cm.get_cmap(COLOR_SCHEMES[color_scheme])(i/4))
    plt.title('脊线图', fontsize=14)

def plot_stream(color_scheme):
    t = np.linspace(0, 10, 100)
    data = np.array([np.sin(t + phase) for phase in [0, 0.5, 1]])
    colors = [plt.cm.get_cmap(COLOR_SCHEMES[color_scheme])(i/2) for i in range(3)]
    plt.stackplot(t, data + 2, labels=['A', 'B', 'C'], colors=colors)
    plt.title('流图', fontsize=14)
    plt.legend()

def plot_andrews_curves(color_scheme):
    data = pd.DataFrame(np.random.randn(10, 4), 
                       columns=['A', 'B', 'C', 'D'])
    pd.plotting.andrews_curves(data, 'A', colormap=plt.cm.get_cmap(COLOR_SCHEMES[color_scheme]))
    plt.title('Andrews曲线', fontsize=14)

def plot_parallel_coordinates(color_scheme):
    np.random.seed(42)
    data = pd.DataFrame({
        'A': np.random.randint(1, 4, 10),
        'B': np.random.randn(10),
        'C': np.random.randn(10),
        'D': np.random.randn(10)
    })
    data['A'] = data['A'].astype(str)
    pd.plotting.parallel_coordinates(
        data, 
        'A',
        color=[plt.cm.get_cmap(COLOR_SCHEMES[color_scheme])(i) for i in np.linspace(0, 1, len(data['A'].unique()))]
    )
    plt.title('平行坐标图', fontsize=14)

def plot_lag_plot(color_scheme):
    np.random.seed(42)
    data = pd.Series(np.random.randn(1000))
    pd.plotting.lag_plot(data, c=plt.cm.get_cmap(COLOR_SCHEMES[color_scheme])(0.6))
    plt.title('滞后图', fontsize=14)

def plot_autocorrelation(color_scheme):
    np.random.seed(42)
    data = pd.Series(np.random.randn(1000))
    pd.plotting.autocorrelation_plot(data)
    plt.gca().lines[0].set_color(plt.cm.get_cmap(COLOR_SCHEMES[color_scheme])(0.6))
    plt.title('自相关图', fontsize=14)

def plot_horizontal_bar(color_scheme):
    categories = ['A', 'B', 'C', 'D', 'E']
    values = np.random.randint(10, 50, size=5)
    colors = [plt.cm.get_cmap(COLOR_SCHEMES[color_scheme])(i/4) for i in range(5)]
    plt.barh(categories, values, color=colors)
    plt.title('水平柱状图', fontsize=14)
    for i, v in enumerate(values):
        plt.text(v + 1, i, str(v), va='center')

def plot_stacked_bar(color_scheme):
    categories = ['A', 'B', 'C', 'D']
    data = np.random.randint(10, 30, size=(3, 4))
    colors = [plt.cm.get_cmap(COLOR_SCHEMES[color_scheme])(i/2) for i in range(3)]
    bottom = np.zeros(4)
    for i, row in enumerate(data):
        plt.bar(categories, row, bottom=bottom, color=colors[i], label=f'Group {i+1}')
        bottom += row
    plt.title('堆叠柱状图', fontsize=14)
    plt.legend()

def plot_grouped_bar(color_scheme):
    categories = ['A', 'B', 'C', 'D']
    data = np.random.randint(10, 30, size=(3, 4))
    x = np.arange(len(categories))
    width = 0.25
    colors = [plt.cm.get_cmap(COLOR_SCHEMES[color_scheme])(i/2) for i in range(3)]
    for i in range(3):
        plt.bar(x + i*width, data[i], width, color=colors[i], label=f'Group {i+1}')
    plt.xticks(x + width, categories)
    plt.title('分组柱状图', fontsize=14)
    plt.legend()

def plot_donut(color_scheme):
    labels = ['A', 'B', 'C', 'D']
    sizes = [15, 30, 45, 10]
    colors = [plt.cm.get_cmap(COLOR_SCHEMES[color_scheme])(i/3) for i in range(4)]
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            pctdistance=0.85, wedgeprops=dict(width=0.5))
    plt.title('环形图', fontsize=14)

def plot_histogram(color_scheme):
    np.random.seed(42)
    data = np.random.normal(0, 1, 1000)
    plt.hist(data, bins=30, color=plt.cm.get_cmap(COLOR_SCHEMES[color_scheme])(0.6),
             alpha=0.7, edgecolor='black')
    plt.title('直方图', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

def plot_area(color_scheme):
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + 2
    plt.fill_between(x, y, color=plt.cm.get_cmap(COLOR_SCHEMES[color_scheme])(0.6), alpha=0.5)
    plt.plot(x, y, color=plt.cm.get_cmap(COLOR_SCHEMES[color_scheme])(0.8))
    plt.title('面积图', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

def plot_stacked_area(color_scheme):
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x) + 2
    y2 = np.cos(x) + 2
    y3 = np.sin(x + np.pi/4) + 2
    colors = [plt.cm.get_cmap(COLOR_SCHEMES[color_scheme])(i/2) for i in range(3)]
    plt.stackplot(x, [y1, y2, y3], colors=colors, labels=['A', 'B', 'C'])
    plt.title('堆叠面积图', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

def plot_hexbin(color_scheme):
    np.random.seed(42)
    x = np.random.normal(0, 1, 1000)
    y = x + np.random.normal(0, 0.5, 1000)
    plt.hexbin(x, y, gridsize=20, cmap=COLOR_SCHEMES[color_scheme])
    plt.colorbar(label='计数')
    plt.title('六边形分箱图', fontsize=14)

def plot_error_bar(color_scheme):
    x = np.arange(5)
    y = np.random.randint(10, 30, 5)
    error = np.random.randint(1, 5, 5)
    plt.errorbar(x, y, yerr=error, fmt='o', color=plt.cm.get_cmap(COLOR_SCHEMES[color_scheme])(0.6),
                capsize=5, capthick=1, ecolor=plt.cm.get_cmap(COLOR_SCHEMES[color_scheme])(0.8))
    plt.title('误差条形图', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

def plot_contour(color_scheme):
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))
    plt.contour(X, Y, Z, cmap=COLOR_SCHEMES[color_scheme])
    plt.colorbar(label='值')
    plt.title('等高线图', fontsize=14)

def plot_gantt(color_scheme):
    tasks = ['Task A', 'Task B', 'Task C', 'Task D']
    starts = [1, 2, 4, 6]
    durations = [2, 3, 2, 1]
    colors = [plt.cm.get_cmap(COLOR_SCHEMES[color_scheme])(i/3) for i in range(4)]
    for i, (task, start, duration) in enumerate(zip(tasks, starts, durations)):
        plt.barh(i, duration, left=start, color=colors[i])
    plt.yticks(range(len(tasks)), tasks)
    plt.xlabel('时间')
    plt.title('甘特图', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

@app.route('/')
def index():
    try:
        color_scheme = request.args.get('color_scheme', 'viridis')
        if color_scheme not in COLOR_SCHEMES:
            color_scheme = 'viridis'
            
        plots = {}
        for plot_name, plot_func in {
            'line': plot_line_chart,
            'bar': plot_bar_chart,
            'scatter': plot_scatter,
            'pie': plot_pie_chart,
            'box': plot_box_chart,
            'heatmap': plot_heatmap,
            'violin': plot_violin,
            '3d_surface': plot_3d_surface,
            'radar': plot_radar,
            'bubble': plot_bubble,
            'density': plot_density,
            'ridgeline': plot_ridgeline,
            'stream': plot_stream,
            'andrews_curves': plot_andrews_curves,
            'parallel_coordinates': plot_parallel_coordinates,
            'lag_plot': plot_lag_plot,
            'autocorrelation': plot_autocorrelation,
            'horizontal_bar': plot_horizontal_bar,
            'stacked_bar': plot_stacked_bar,
            'grouped_bar': plot_grouped_bar,
            'donut': plot_donut,
            'histogram': plot_histogram,
            'area': plot_area,
            'stacked_area': plot_stacked_area,
            'hexbin': plot_hexbin,
            'error_bar': plot_error_bar,
            'contour': plot_contour,
            'gantt': plot_gantt
        }.items():
            try:
                plots[plot_name] = generate_plot(plot_func, color_scheme)
            except Exception as e:
                print(f'Error generating {plot_name} plot: {str(e)}')
                plots[plot_name] = ''
        
        return render_template('index.html', plots=plots, color_schemes=list(COLOR_SCHEMES.keys()), current_scheme=color_scheme)
    except Exception as e:
        print(f'Error in index route: {str(e)}')
        return 'An error occurred while generating plots', 500

if __name__ == '__main__':
    app.run(debug=True, threaded=True)