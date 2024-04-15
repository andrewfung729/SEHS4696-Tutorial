import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import fontManager

fontManager.addfont("NotoSansTC-Regular.ttf")
# mpl.rc("font", family="Noto Sans TC", size=14)

list1 = [78, 80, 79, 81, 91, 95, 96]
X = np.arange(1, 8)
fig = plt.figure(num=1, figsize=(8, 8))

ax = fig.add_subplot(111)

ax.set_xlim([1, 7.1])
ax.set_xticks(
    np.linspace(1, 7, 7),
    [
        "星期一",
        "星期二",
        "星期三",
        "星期四",
        "星期五",
        "星期六",
        "星期日",
    ],
)
ax.set_xlabel("星期", fontproperties="Noto Sans TC", fontsize=12)

ax.set_ylim([70, 100])
ax.set_yticks(np.linspace(70, 100, 4), ["70kg", "80kg", "90kg", "100kg"])
ax.set_ylabel("銷量", fontproperties="Noto Sans TC", fontsize=12)

ax.tick_params(axis="x", labelsize=12, labelrotation=30, labelfontfamily="Noto Sans TC")
ax.set_title("上星期銷量", fontproperties="Noto Sans TC", fontsize=16)

spines = ax.spines
spines["bottom"].set_linewidth(3)
spines["left"].set_color("darkblue")
spines[["top", "right"]].set_visible(False)

ax.plot(X, list1, "r-.d")

ax.legend(
    labels=["Apple", "Banana"],
    loc="upper left",
    labelspacing=2,
    handlelength=4,
    fontsize=14,
    shadow=True,
)

# mark the min and max value
min_index = list1.index(min(list1))
min_value = min(list1)
max_index = list1.index(max(list1))
max_value = max(list1)
ax.text(max_index, max_value, f"max: {max_value}", fontsize=12)
ax.annotate(
    f"min: {min_value}",
    xy=(min_index + 1, min_value),
    xytext=(min_index + 1 + 0.5, min_value - 1.5),
    arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=5),
    fontsize=12,
)

plt.show()
