class PIDcontroller:
    def __init__(self):

        self.p_p = 0.1
        self.p_i = 0.00009
        self.p_d = 3.5
        self.prev_CTE = 0.0
        self.CTE_total = 0.0

        self.p_error = self.d_error = self.i_error = 0.0



    def process(self, CTE):

        self.CTE_total += abs(CTE)
        
        P = -self.p_p * CTE
        I = -self.p_i * self.CTE_total
        D = -self.p_d * (CTE - self.prev_CTE)

        self.prev_CTE = CTE

        return P + I + D