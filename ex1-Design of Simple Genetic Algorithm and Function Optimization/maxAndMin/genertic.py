import math, random, sys
from matplotlib import pyplot as plt


class Population:
    # 种群设计
    def __init__(self, size, chrom_size, cp, mp, gen_max, mark):
        # 种群信息
        self.individuals = []  # 个体集合
        self.fitness = []  # 个体适应度集
        self.selector_probability = []  # 个体选择概率
        self.fine_individuals = []  # 优秀个体集合
        self.new_individuals = []  # 新一代个体集合
        self.indiv_fits = []  # 个体及其适应度对应组合列表

        self.best = []

        self.size = size  # 种群所包含的个体数
        self.chromosome_size = chrom_size  # 个体的染色体长度
        self.crossover_probability = cp  # 个体之间的交叉概率
        self.mutation_probability = mp  # 个体之间的变异概率
        self.mark = mark  # 标记求函数最大值还是最小值

        self.generation_max = gen_max  # 种群进化的最大世代数
        self.age = 0  # 种群当前所处世代

        # 随机产生初始个体集，并将新一代个体、适应度、选择概率等集合以 0 值进行初始化
        v = 2 ** self.chromosome_size - 1
        for i in range(self.size):
            self.individuals.append(random.randint(0, v))
            self.new_individuals.append(0)
            self.fitness.append(0)
            self.selector_probability.append(0)

    ### 给予轮盘赌博机的选择 ###

    # 解码
    def decode(self, interval, chromosome):
        '''将一个染色体 chromosome 映射为区间 interval 之内的数值'''
        d = interval[1] - interval[0]
        n = float(2 ** self.chromosome_size - 1)
        return (interval[0] + chromosome * d / n)

    # 适应度函数
    def fitness_func(self, chrom):
        '''适应度函数，可以根据个体的两个染色体计算出该个体的适应度'''
        interval = [0, 2 * math.pi]
        x = self.decode(interval, chrom)
        if self.mark == 'min':
            return -(x * math.sin(x) + 1)
        elif self.mark == 'max':
            return x * math.sin(x) + 1
        else:
            print('参数有误！程序终止。。。')
            sys.exit()

    # 评估函数
    def evaluate(self):
        '''用于评估种群中的个体集合 self.individuals 中各个个体的适应度'''
        sp = self.selector_probability
        for i in range(self.size):
            self.fitness[i] = self.fitness_func(self.individuals[i])  # 将计算结果保存在 self.fitness 列表中
            self.indiv_fits.append([self.individuals[i], self.fitness[i]])

        ft_sum = sum(self.fitness)
        for i in range(self.size):
            sp[i] = self.fitness[i] / float(ft_sum)  # 得到各个个体生存概率

        for i in range(1, self.size):
            sp[i] = sp[i] + sp[i - 1]  # 需要将个体的生存概率进行叠加，从而计算出各个个体的选择概率

    # 轮盘赌博机 （选择）
    def select(self):
        (t, i) = (random.random(), 0)
        for p in self.selector_probability:
            if p > t:
                break
            i = i + 1
        return i

    # 按 self.fitness 值对 self.indiv_fits 进行降序排序
    def takeSecond(self, elem):
        return elem[1]

    def Sort(self):
        self.indiv_fits.sort(key=self.takeSecond, reverse=True)

    # 复制一半优秀个体
    def copy_indiv(self):
        for i in range(self.size // 2):
            self.new_individuals[i] = self.indiv_fits[i][0]

    # 交叉
    def cross(self, chrom1, chrom2):
        p = random.random()  # 随机概率
        n = 2 ** self.chromosome_size - 1
        if chrom1 != chrom2 and p < self.crossover_probability:
            t = random.randint(1, self.chromosome_size - 1)  # 随机选择一点
            mask = n << t  # 左移
            (r1, r2) = (chrom1 & mask, chrom2 & mask)
            mask = n >> (self.chromosome_size - t)
            (l1, l2) = (chrom1 & mask, chrom2 & mask)
            (chrom1, chrom2) = (r1 + l2, r2 + l1)
        return (chrom1, chrom2)

    # 变异
    def mutate(self, chrom):
        p = random.random()
        if p < self.mutation_probability:
            t = random.randint(1, self.chromosome_size)
            mask1 = 1 << (t - 1)
            mask2 = chrom & mask1
            if mask2 > 0:
                chrom = chrom & (~mask2)  # ~ 按位取反运算符：对数据的每个二进制位取反,即把1变为0,把0变为1
            else:
                chrom = chrom ^ mask1  # ^ 按位异或运算符：当两对应的二进位相异时，结果为1
        return chrom

    # 进化过程
    def evolve(self):
        indvs = self.individuals
        new_indvs = self.new_individuals
        # 计算适应度及选择概率
        self.evaluate()
        self.Sort()
        self.copy_indiv()
        i = self.size // 2
        while True:
            # 选择两个个体，进行交叉与变异，产生新的种群
            idv1 = self.select()
            idv2 = self.select()
            # 交叉
            idv1_x = indvs[idv1]
            idv2_x = indvs[idv2]
            (idv1_x, idv2_x) = self.cross(idv1_x, idv2_x)
            # 变异
            (idv1_x, idv2_x) = (self.mutate(idv1_x), self.mutate(idv2_x))
            (new_indvs[i], new_indvs[i + 1]) = (idv1_x, idv2_x)  # 将计算结果保存于新的个体集合self.new_individuals中
            # 判断进化过程是否结束
            i = i + 2  # 循环self.size/2次，每次从self.individuals 中选出2个
            if i >= self.size - 1:
                self.new_individuals[self.size - 1] = self.individuals[self.size - 1]
                break

        # 更新换代：用种群进化生成的新个体集合 self.new_individuals 替换当前个体集合
        for i in range(self.size):
            self.individuals[i] = self.new_individuals[i]

    def run(self):
        '''
        根据种群最大进化世代数设定了一个循环。
        在循环过程中，调用 evolve 函数进行种群进化计算，并输出种群的每一代的个体适应度最大值、平均值和最小值。
        '''
        if self.mark == 'min':
            # print("--------------------------------------")
            print("函数最小值迭代过程:")
            # print("--------------------------------------")
        elif self.mark == 'max':
            # print("--------------------------------------")
            print("函数最大值迭代过程:")
            # print("--------------------------------------")
        for i in range(self.generation_max):
            self.evolve()
            if i % 10 == 0:
                if self.mark == 'max':
                    print(i, max(self.fitness))
                elif self.mark == 'min':
                    print(i, -max(self.fitness))
            if self.mark == 'max':
                self.best.append(max(self.fitness))
            elif self.mark == 'min':
                self.best.append(-max(self.fitness))


if __name__ == '__main__':
    # 种群的个体数量为 10，染色体长度为 24，交叉概率为 0.7，变异概率为 0.15,进化最大世代数为 150
    pop_max = Population(6, 10, 0.7, 0.1, 200, 'max')
    pop_max.run()
    pop_min = Population(6, 10, 0.7, 0.1, 200, 'min')
    pop_min.run()
    print()
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print("函数最大值为:{0} \n函数最小值为:{1}".format(pop_max.best[pop_max.generation_max - 1],
                                                       pop_min.best[pop_min.generation_max - 1]))
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    x = range(pop_max.generation_max)
    plt.figure()
    plt.plot(x, pop_max.best, c='r')
    plt.plot(x, pop_min.best, c='b')
    plt.show()
