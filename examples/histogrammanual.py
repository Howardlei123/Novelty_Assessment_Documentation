import matplotlib.pyplot as plt

bins = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]  
frequency = [3,18,16,7,6,2,3,3,1,1]  

plt.bar(range(len(frequency)), frequency, width=1, edgecolor='black', color = 'purple',align='edge')

# 使用bins数组的值作为x轴的刻度标签
plt.xticks(range(len(frequency)), bins)

plt.title('Predefined Frequency Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

plt.show()