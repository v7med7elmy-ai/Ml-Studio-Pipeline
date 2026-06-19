import matplotlib.pyplot as plt
import seaborn as sns

def plot_line(df, y, x=None, hue=None):
    x_data  = df.index if x is None else df[x]
    x_label = 'Index'  if x is None else x

    fig, ax = plt.subplots(figsize=(10, 5))

    if hue and hue != "None":
        for val in df[hue].unique():
            mask = df[hue] == val
            ax.plot(x_data[mask], df[y][mask], label=str(val), linewidth=1.8)
        ax.legend(title=hue)
    else:
        ax.plot(x_data, df[y], linewidth=1.8)

    ax.set_title(f'Line Plot — {y}', fontsize=13, pad=10)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y)
    plt.tight_layout()
    return fig  # رجعنا الـ fig عشان الـ GUI يقراها

def plot_scatter(df, x, y, hue=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # تحويل "None" لنص حقيقي يفهمه السيبورن
    hue_param = None if hue == "None" else hue

    sns.scatterplot(
        data=df, x=x, y=y,
        hue=hue_param, palette='tab10',
        alpha=0.75, s=60, ax=ax
    )

    ax.set_title(f'Scatter Plot — {x} vs {y}', fontsize=13, pad=10)
    plt.tight_layout()
    return fig  # رجعنا الـ fig هنا كمان

def plot_box(df, y, x=None, hue=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # تظبيط البارامترات بالأمان للـ boxplot
    x_param = None if x == "None" else x
    hue_param = None if hue == "None" else hue

    sns.boxplot(
        data=df, x=x_param, y=y,
        hue=hue_param, palette='Set2',
        linewidth=1.2,
        flierprops=dict(marker='o', markersize=3, alpha=0.5),
        ax=ax
    )

    title = f'Box Plot — {y}'
    if x_param:
        title += f' grouped by {x_param}'
    ax.set_title(title, fontsize=13, pad=10)
    plt.tight_layout()
    return fig  # رجعنا الـ fig