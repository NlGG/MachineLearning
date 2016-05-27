import numpy as np
import math 

def rosenbrock(x):
    z = 0
    for i in range(1, len(x)):
        z += 100*pow(x[0] - x[i], 2) + pow(1 - x[i], 2)
    return z

class SA:
    #引数にはN:変数の数、x:変数の下限と上限、init_T:温度の初期値、をとる。
    def __init__(self, N=50, x=[-2.048, 2.048], init_T=100):
        self.x = np.array([np.random.uniform(x[0], x[1]) for i in range(N)])
        self.T = init_T
        self.N = N
        self.index = np.array([i for i in range(N)])
    
    #annealingで使うための関数である。
    def copy_x(self, change):
        x = np.zeros(self.N)
        for i in range(self.N):
            if i != change:
                x[i] = self.x[i]
            else:
                a = np.random.choice([0, 1])
                if a == 0:
                    x[change] += 0.001*np.random.uniform(0, 2)
                else:
                    x[change] -= 0.001*np.random.uniform(0, 2)
        return x
      
    def annealing(self):
        
        change = np.random.choice(self.index)
        #配列xのうち変更するインデックスを決定する。
        
        x = self.copy_x(change)
        #x[change]を正または負の方向にわずかに動かす。
            
        energy = self.accept(x) - self.accept(self.x)
        #変更前と変更後のエネルギー差。
       
        #エネルギーが低くなれば値を更新、そうでなくとも一定の確率(Tに依存)で更新。
        if energy < 0:
            self.x = x　
        else:
            p = pow(math.e, -energy/self.T)
            b = np.random.choice([0, 1], p=[p, 1-p])
            if b == 0:
                self.x = x
    
    #エネルギーの冷却
    def cooling(self):
        self.T = self.T*0.95
    
    #rosenbrock関数の計算。
    def accept(self, x):    
        cal = rosenbrock(x)
        return cal
    
    #n回繰り返す。
    def simulate(self, n):
        self.log = np.array([])
        for i in range(n):
            self.annealing()
            self.cooling()
            self.log = np.append(self.log, self.accept(self.x))
        return self.log[-1]

#粒子を表す。
class particle:
    #引数の意味はx:粒子の位置、v:粒子の速度、alpha,beta,ganmaは速度更新で重みの調整に使われる。
    def __init__(self, x, v, omega, alpha, beta, ganma):
        self.x = x
        self.v = v
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.ganma = ganma
        self.lbest_x = x
        self.lbest = 10000
    
    #その粒子のローカルベストを返す関数。   
    def local_best(self):
        cal = rosenbrock(self.x)
        if self.lbest > cal:
            self.lbest = cal
            self.lbest_x = self.x
    
    #vの値の更新。
    def renew_v(self, gbest_x):
        r_1 = np.random.uniform()
        r_2 = np.random.uniform()
        self.v = self.ganma*(self.omega*self.v + (self.lbest_x - self.x)*self.alpha*r_1 + (gbest_x - self.x)*self.beta*r_2)
    #xの値の更新。
    def renew_x(self):
        self.x = self.x + self.v

#PSO全体のクラス。
class PSO:
    #引数の意味はN:変数の数、pN:粒子の数、limit_x:変数（粒子の位置）の下限と上限、残りはそれぞれ対応する変数の下限と上限（値は経験的に決めた）。
    def __init__(self, N=50, pN=100, limit_x=[-2.048, 2.048], limit_v=[0.0, 0.01], limit_omega=[0.8, 1.0], limit_alpha=[0.0, 1.0], limit_beta=[0.0, 1.0], limit_ganma=[0.8, 1.0]):
        #Alpha is Attenuation coefficient.
        #Omega is Convergence factor.
        #N is the number of iteration. Exit condition is given by the number of iteration.
        self.N = N
        self.pN = pN
        self.limit_x = limit_x
        self.limit_v = limit_v
        self.limit_omega = limit_omega
        self.limit_alpha = limit_alpha
        self.limit_ganma = limit_ganma
        self.limit_beta = limit_beta
        self.GBest = 10000
        self.GBest_x = np.array([np.random.uniform(limit_x[0], limit_x[1]) for i in range(N)])
    
    #以下、initialize_?関数は?の初期化を行う。
    def initialize_x(self):   
        x = np.array([np.random.uniform(self.limit_x[0], self.limit_x[1]) for i in range(self.N)])
        return x
    
    def initialize_v(self):
        v = np.array([np.random.uniform(self.limit_v[0], self.limit_v[1]) for i in range(self.N)])
        return v
        
    def initialize_omega(self):
        omega = np.array([np.random.uniform(self.limit_omega[0], self.limit_omega[1]) for i in range(self.N)])
        return omega
        
    def initialize_alpha(self):
        alpha = np.array([np.random.uniform(self.limit_alpha[0], self.limit_alpha[1]) for i in range(self.N)])
        return alpha
    
    def initialize_beta(self):
        beta = np.array([np.random.uniform(self.limit_beta[0], self.limit_beta[1]) for i in range(self.N)])
        return beta
    
    def initialize_ganma(self):
        ganma = np.array([np.random.uniform(self.limit_ganma[0], self.limit_ganma[1]) for i in range(self.N)])
        return ganma
     
    #粒子をセットする。配列に粒子オブジェクトを格納する。
    def set_particles(self):
        particles = np.array([])
        for i in range(self.pN):
            particles = np.append(particles, particle(self.initialize_x(), self.initialize_v(), self.initialize_omega(), self.initialize_alpha(), self.initialize_beta(), self.initialize_ganma()))
            particles[i].local_best()
            if particles[i].lbest < self.GBest:
                self.GBest = particles[i].lbest 
                self.GBest_x = particles[i].lbest_x
            particles[i].renew_v(self.GBest_x)
            self.particles = particles
    
    #n回シミュレートする。GBestはグローバルベストを示す。
    def simulate(self, n):
        self.set_particles()
        for t in range(n):
            for i in range(self.pN):
                self.particles[i].renew_x()
                self.particles[i].local_best()
                if self.particles[i].lbest < self.GBest:
                    self.GBest = self.particles[i].lbest 
                    self.GBest_x = self.particles[i].lbest_x
                self.particles[i].renew_v(self.GBest_x)
        return self.GBest, self.GBest_x
    
    #1回シミュレートする。
    def one_simulate(self, n):
        for t in range(n):
            for i in range(self.pN):
                self.particles[i].renew_x()
                self.particles[i].local_best()
                if self.particles[i].lbest < self.GBest:
                    self.GBest = self.particles[i].lbest 
                    self.GBest_x = self.particles[i].lbest_x
                self.particles[i].renew_v(self.GBest_x)
        return self.GBest, self.GBest_x