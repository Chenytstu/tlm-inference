import math
import numpy as np

from Communication.api import *
from Configs.constant import *
from Linear.Mult import Mult
from role import *
import Configs.communication as config

def show1(x):
    for i in x:
        print(i, end=" ")
    print()

def show(x):
    for i in x:
        for j in i:
            print(j, end=" ")
        print()

def gen_mask(row: int, column: int):
    result = []
    mul_tmp = 0
    for _ in range(row - 1):
        tmp = []
        mul_tmp1 = 1
        for __ in range(column):
            tmp.append(famcfrac(np.random.uniform(0, 2)))
            mul_tmp1 *= tmp[__]
        mul_tmp += mul_tmp1
        result.append(tmp)
    mul_tmp = 1 - mul_tmp
    tmp = []
    mul_tmp1 = 1
    for _ in range(column):
        tmp.append(famcfrac(np.random.uniform(0, 2)))
        if _ != column - 1:
            mul_tmp1 *= tmp[_]
    result.append(tmp)
    result[row - 1][column - 1] = famcfrac(mul_tmp / mul_tmp1);
    return np.asarray(result, dtype=object)
    

class NonLinear:
    def __init__(self, party: int):
        assert(party == Alice or party == Bob)
        self._party = party
        self._mult = Mult(party)
        
    def mul2add(self, x, rand1, rand2):
        if (self._party == Alice):
            mask_x = x * rand1
            send(mask_x, port=config.default_port_1)
        else:
            mask_x = x * rand2
            mask_x_remote = recv(port=config.default_port_1)
            send(mask_x, port=config.default_port_2)
            result = mask_x_remote * rand1 * x
            return result
        mask_x_remote = recv(port=config.default_port_2)
        result = mask_x_remote * rand2 * x
        return result
    
    def inverse_square(self, x, a, b, c, iter: int=10):
        rand = np.random.uniform(np.sqrt(.5), np.sqrt(2))
        inverse = None
        if self._party == Alice:
            send(rand * x, port=config.default_port_1)
            inverse = recv(port=config.default_port_2)
        else:
            remote = recv(port=config.default_port_1)
            alpha = int(x * rand + remote)
            inverse_ = math.pow(2, (int) (alpha / -2))
            inverse = famefrac(np.random.uniform(0, inverse_))
            send(inverse_ - inverse, port=config.default_port_2)
        tmp = inverse
        for _ in range(iter):
            tmp1 = tmp
            
            inverse2 = self._mult.vector_multipication(inverse, inverse, a, b, c)
            iter_ = self._mult.vector_multipication(x, inverse2, a, b, c)
            iter__ = self._mult.vector_multipication(inverse, iter_ * -1 + 1.5, a, b, c)
            tmp = iter__ * .5
            
            inverse = tmp1
        return inverse
    
    def layerNorm(self, x, a, b, c, gamma=1, beta=0, iter: int=10):
        avg = x.sum() / len(x)
        sigma = x - avg
        var_ = self._mult.vector_multipication(sigma, sigma, a, b, c)
        var = var_.sum() / len(var_)
        inverse_var = self.inverse_square(var, a, b, c, iter)
        norm = self._mult.vector_multipication(sigma, inverse_var, a, b, c)
        return norm * gamma + beta
        
    def softmax(self, x, rand1, rand2):
        rand = famefrac(np.random.uniform(0, 1))
        exp_x = []
        max_ = x[0]
        for i in x:
            if max_ < i:
                max_ = i
        for i in x - max_:
            exp_x.append(i.exp() * rand)
        exp_x = np.asarray(exp_x, dtype=object)
        exp_x_addshare = self.mul2add(exp_x, rand1, rand2)
        sum_ = exp_x_addshare.sum()
        sum_remote = None
        if (self._party == Alice):
            send(sum_, port=config.default_port_1)
            sum_remote = recv(port=config.default_port_2)
        else:
            sum_remote = recv(port=config.default_port_1)
            send(sum_, port=config.default_port_2)
        return exp_x_addshare / (sum_ + sum_remote)
    
    def gelu(self, x, a, b, c, rand1, rand2):
        x2 = self._mult.vector_multipication(x, x, a, b, c)
        x3 = self._mult.vector_multipication(x, x2, a, b, c)
        alpha =  (x + x3 * 0.0044715) * famcfrac(math.sqrt(2. / math.pi) * 2)
        rand = famefrac(np.random.uniform(0, 1))
        rand = famefrac(1)
        theta = []
        for i in alpha:
            theta.append(i.exp() * rand)
        theta = np.asarray(theta, dtype=object)
        rand_addshare = self.mul2add(rand, rand1, rand2)
        theta_addshare = self.mul2add(theta, rand1, rand2)
        omega_local = theta_addshare + rand_addshare
        if self._party == Alice:
            send(omega_local, port=config.default_port_1)
        else:
            omega_remote = recv(port=config.default_port_1)
            omega = omega_local + omega_remote
            send(omega, port=config.default_port_2)
            beta = (theta_addshare - rand_addshare) / omega + .5
            return self._mult.vector_multipication(beta, x, a, b, c) * .5
        omega_remote = recv(port=config.default_port_2)
        beta = (theta_addshare - rand_addshare) / omega_remote + .5
        return self._mult.vector_multipication(beta, x, a, b, c) * .5
        

def real_softmax(x):
    exp_x = []
    for i in x:
        exp_x.append(i.exp())
    exp_x = np.asarray(exp_x, dtype=object)
    return exp_x / exp_x.sum()

def real_gelu(x):
    import math
    alpha = (x + x * x * x * 0.0044715) * famcfrac(math.sqrt(2. / math.pi) * 2)
    exp_alpha = []
    for i in alpha:
        exp_alpha.append(i.exp())
    exp_alpha = np.asarray(exp_alpha, dtype=object)
    beta = (exp_alpha - 1) / (exp_alpha + 1) + 1
    return beta * x * .5

def real_layerNorm(x):
    avg = x.sum() / len(x)
    sigma = x - avg
    var_ = sigma * sigma
    var = sum(var_) / len(var_)
    inverse_var = 1. / math.sqrt(var)
    return sigma * inverse_var

def real_inverse_square(x):
    result = []
    for i in x:
        result.append(1 / math.sqrt(i))
    return np.asarray(result, dtype=object)

if __name__ == "__main__":
    dim = 4
    mask = gen_mask(2, 2)
    x = []
    for j in range(dim):
        x.append(famcfrac(np.random.uniform(-1, 1)))
    x = np.asarray(x, dtype=object)
    party = int(sys.argv[1])
    non_linear = NonLinear(party)
    # if (party == Alice):
    #     send((mask[0][1], mask[1][1]))
    #     res = non_linear.mul2add(x, mask[0][0], mask[1][0])
    #     send((x, res))
    # else:
    #     mask[0][1], mask[1][1] = recv()
    #     res = non_linear.mul2add(x, mask[0][1], mask[1][1])
    #     remote_x, res_remote = recv()
    #     show1(x * remote_x - res - res_remote)
    # if (party == Alice):
    #     send((mask[0][1], mask[1][1]))
    #     res = non_linear.softmax(x, mask[0][0], mask[1][0])
    #     send((x, res))
    # else:
    #     mask[0][1], mask[1][1] = recv()
    #     res = non_linear.softmax(x, mask[0][1], mask[1][1])
    #     remote_x, res_remote = recv()
    #     show1(res + res_remote - real_softmax(x + remote_x))
    # if (party == Alice):
    #     send((mask[0][1], mask[1][1]))
    #     res = non_linear.gelu(x, mask[0][0], mask[1][0], 1, 1, 2)
    #     send((x, res))
    # else:
    #     mask[0][1], mask[1][1] = recv()
    #     res = non_linear.gelu(x, mask[0][1], mask[1][1], 1, 1, 2)
    #     remote_x, res_remote = recv()
    #     show1(res + res_remote - real_gelu(x + remote_x))
    # x = famcfrac(np.random.uniform(0, 2))
    # if (party == Alice):
    #     res = non_linear.inverse_square(x, 1, 1, 2)
    #     send((x, res))
    # else:
    #     res = non_linear.inverse_square(x, 1, 1, 2)
    #     remote_x, res_remote = recv()
    #     print((res + res_remote - 1 / (math.sqrt(x + remote_x))) * 10 ** 13)avg = x.sum() / len(x)
    avg = x.sum() / len(x)
    sigma = x - avg
    var_ = sigma * sigma
    var = sum(var_) / len(var_)
    non_linear.inverse_square(var, 1, 1, 2)
    # if (party == Alice):
    #     res_iq = non_linear.inverse_square(var, 1, 1, 2)
    #     res_ln = non_linear.layerNorm(x, 1, 1, 2)
    #     send((x, res_ln, var, res_iq))
    # else:
    #     res_iq = non_linear.inverse_square(var, 1, 1, 2)
    #     res_ln = non_linear.layerNorm(x, 1, 1, 2)
    #     remote_x, res_remote_ln, var_remote, res_remote_iq= recv()
    #     print(var_remote + var)
    #     print(var_remote, var)
    #     print(res_iq + res_remote_iq)
    #     print(res_iq + res_remote_iq - 1. / math.sqrt(var + var_remote))
    #     print()
        
    #     show1(remote_x + x)
    #     show1(remote_x)
    #     show1(x)
    #     show1(res_ln + res_remote_ln)
    #     show1(res_ln + res_remote_ln - real_layerNorm(x + remote_x))
    