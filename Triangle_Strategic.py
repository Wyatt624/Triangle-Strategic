import time
import pandas as pd
import numpy as np
import bottleneck as bn
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import argrelextrema
from scipy.stats import linregress
import mplfinance as mpf
from scipy.ndimage.interpolation import shift

class BackTrader():
    def __init__(self,start=[2022,1,30],end=[2022,12,30],oraccount=10000,spreads=0.02,Hold_time_set=[False,7],Profit_Set=[False,0.20,-0.20]):
        # 回测时间段
        self.start_time=datetime.date(start[0],start[1],start[2])
        self.end_time=datetime.date(end[0],end[1],end[2])

        # 本金
        self.ora=oraccount

        self.spreads=spreads  # 滑率

        # # 持股天数机制
        # self.Hold_Time_set=Hold_time_set

        # 利润止盈止损机制
        self.Profit_Set=Profit_Set

    def Load_Data(self,path='E:/Quant_stock_data/Stock_day_data/HSAday.pkl',single_stock=[False,'']):
        self.path=path
        hsa=pd.read_pickle(path)
        hsa=hsa.set_index('Date')
        hsa=hsa.groupby('Code').apply(lambda x: x[self.start_time:self.end_time])
        hsa=hsa.reset_index(level='Date')
        hsa=hsa.reset_index(drop=True)
        self.hsa=hsa
        self.single_stock=single_stock
        if single_stock[0]:
            self.hsa=self.hsa.loc[self.hsa['Code']==single_stock[1]]
    
    def Strategic_Triangle(self,neighbor=5,Triangle_point=3,rvalue=0,slope_para=[0.05,0.5,0.5,1,1],Strategic=1):
        # Calculate the peaks and floors! 比较领域
        def peak_find(x,neighbor):
            indexs=x.index
            x=x.reset_index(drop=True)
            peaks=argrelextrema(x.to_numpy(),np.greater,order=neighbor)
            x.loc[x.index.isin(peaks[0])]=1
            x.loc[~x.index.isin(peaks[0])]=0
            return pd.DataFrame(x.to_list(),index=indexs,columns=['Peak_Sign'])
        def floor_find(x,neighbor):
            indexs=x.index
            x=x.reset_index(drop=True)
            peaks=argrelextrema(x.to_numpy(),np.less,order=neighbor)
            x.loc[x.index.isin(peaks[0])]=-1
            x.loc[~x.index.isin(peaks[0])]=0
            return pd.DataFrame(x.to_list(),index=indexs,columns=['Floor_Sign'])

        self.hsa['Peak_Sign']=self.hsa.groupby('Code')['High_adjusted'].apply(peak_find,neighbor)
        self.hsa['Peak_Price']=self.hsa['Peak_Sign'].copy()
        self.hsa['Peak_Price']=self.hsa['Peak_Price'].mask(self.hsa['Peak_Sign']!=0,self.hsa['High_adjusted'])
        self.hsa['Floor_Sign']=self.hsa.groupby('Code')['Low_adjusted'].apply(floor_find,neighbor)
        self.hsa['Floor_Price']=self.hsa['Floor_Sign'].copy()
        self.hsa['Floor_Price']=self.hsa['Floor_Price'].mask(self.hsa['Floor_Sign']!=0,self.hsa['Low_adjusted'])

        def slope_cal(y,r,slope):
            t =[s.strftime('%Y-%m-%d') for s in y.index]
            t = pd.Series(t)
            t = pd.to_datetime(t)
            t=(t-t[0]).astype('timedelta64[D]')
            res=linregress(t,y)
            if abs(res.rvalue) >= r:
                if slope:
                    return res.slope
                else:
                    return y[-1]
            else:
                return np.nan
            
        # Calculate the slope of each  每Triangle_point个点计算一次斜率
        # slope: True-->获取斜率，False-->获取最近一个峰值或谷底
        def rolling_Peak(x,Triangle_point,slope):
            indexs=x.index
            x=x.set_index('Date')
            if slope:
                col='Peak_Slope'
            else:
                col='Peak_nearest' 
            return pd.DataFrame(x.rolling(Triangle_point).apply(slope_cal,args=(rvalue,slope,)).to_numpy(),index=indexs,columns=[col])
        def rolling_Floor(x,Triangle_point,slope):
            indexs=x.index
            x=x.set_index('Date')
            if slope:
                col='Floor_Slope'
            else:
                col='Floor_nearest'
            return pd.DataFrame(x.rolling(Triangle_point).apply(slope_cal,args=(rvalue,slope,)).to_numpy(),index=indexs,columns=[col])

        ps=self.hsa.loc[self.hsa['Peak_Sign']==1,]
        self.hsa['Peak_Slope'],self.hsa['Peak_nearest']=0,0
        self.hsa.loc[self.hsa['Peak_Sign']==1,'Peak_Slope']=ps.groupby('Code')[['Date','High_adjusted']].apply(rolling_Peak,Triangle_point=Triangle_point,slope=True)
        self.hsa.loc[self.hsa['Peak_Sign']==1,'Peak_nearest']=ps.groupby('Code')[['Date','High_adjusted']].apply(rolling_Peak,Triangle_point=Triangle_point,slope=False)

        fs=self.hsa.loc[self.hsa['Floor_Sign']==-1,]
        self.hsa['Floor_Slope'],self.hsa['Floor_nearest']=0,0
        self.hsa.loc[self.hsa['Floor_Sign']==-1,'Floor_Slope']=fs.groupby('Code')[['Date','Low_adjusted']].apply(rolling_Floor,Triangle_point=Triangle_point,slope=True)
        self.hsa.loc[self.hsa['Floor_Sign']==-1,'Floor_nearest']=fs.groupby('Code')[['Date','Low_adjusted']].apply(rolling_Floor,Triangle_point=Triangle_point,slope=False)

        # 标记空区间，并解决峰值谷底出现导致的信号识别问题！！！
        # 数据滞后一天进行识别箱体，即当天出现峰值和谷底不会加入计算当时的箱体状态！ 当天的箱体识别使用的是昨天之前的峰值谷底~
        self.hsa['Peak_Slope_shift']=self.hsa.groupby('Code')['Peak_Slope'].transform(Fill,Shift=True)
        self.hsa['Peak_nearest_shift']=self.hsa.groupby('Code')['Peak_nearest'].transform(Fill,Shift=True)
        self.hsa['Floor_Slope_shift']=self.hsa.groupby('Code')['Floor_Slope'].transform(Fill,Shift=True)
        self.hsa['Floor_nearest_shift']=self.hsa.groupby('Code')['Floor_nearest'].transform(Fill,Shift=True)
        # 未滞后
        self.hsa['Peak_Slope']=self.hsa.groupby('Code')['Peak_Slope'].transform(Fill,Shift=False)
        self.hsa['Peak_nearest']=self.hsa.groupby('Code')['Peak_nearest'].transform(Fill,Shift=False)
        self.hsa['Floor_Slope']=self.hsa.groupby('Code')['Floor_Slope'].transform(Fill,Shift=False)
        self.hsa['Floor_nearest']=self.hsa.groupby('Code')['Floor_nearest'].transform(Fill,Shift=False)
        
        # self.hsa.to_pickle('E:/Quant_stock_data/Strategic_Triangle/HSAday_Triangle.pkl')

        # Strategic_Triangle
        # 判断收敛三角形的类型:
        self.hsa['Normal Triangle']=np.nan
        # 正，向上，向下收敛三角形均是使用价格突破支撑位和阻力位形成买卖信号，故使用滞后一期数据！
        # 正收敛三角形: -slope[0]<peak<0<floor<slope[0]
        self.hsa['Normal Triangle']=self.hsa['Normal Triangle'].mask((-slope_para[0]<self.hsa['Peak_Slope_shift'])&(self.hsa['Peak_Slope_shift']<0)&(0<self.hsa['Floor_Slope_shift'])&(self.hsa['Floor_Slope_shift']<slope_para[0]),0)
        # 向上收敛三角形: 向上收敛，0<slope[1]<peak<floor
        self.hsa['Normal Triangle']=self.hsa['Normal Triangle'].mask((slope_para[1]<self.hsa['Peak_Slope_shift'])&(self.hsa['Peak_Slope_shift']<self.hsa['Floor_Slope_shift']),-1)
        # 向下收敛三角形: 向下收敛，floor斜率为负，且peak<floor<slope[2]
        self.hsa['Normal Triangle']=self.hsa['Normal Triangle'].mask((self.hsa['Peak_Slope_shift']<self.hsa['Floor_Slope_shift'])&(self.hsa['Floor_Slope_shift']<-slope_para[2]),1)

        # 增加发散三角形，因为该特殊三角形出现时，一般为骤升或骤降，可立即进行买入卖出操作，故不需要使用滞后数据！
        self.hsa['Special Triangle']=np.nan
        # 向上发散三角形：向上发散，释放能量，slope[0]<<slope[3]<peak and slope[0]<floor
        self.hsa['Special Triangle']=self.hsa['Special Triangle'].mask((slope_para[3]<self.hsa['Peak_Slope'])&(slope_para[0]<self.hsa['Floor_Slope']),-2)
        # 向下发散三角形：向下发散，释放能量，floor<-slope[4]<<-slope[0] and peak<-slope[0]
        self.hsa['Special Triangle']=self.hsa['Special Triangle'].mask((self.hsa['Floor_Slope']<-slope_para[4])&(self.hsa['Peak_Slope']<-slope_para[0]),2)

        # # 没有计算出箱体的，提取上一个箱体
        # def Fill_Triangle(x):
        #     x2=x.to_list()
        #     for i in range(1,len(x2),1):
        #         if np.isnan(x2[i]):
        #             x2[i]=x2[i-1]
        #     return x2
        # self.hsa['Normal Triangle']=self.hsa.groupby('Code')['Normal Triangle'].transform(Fill_Triangle)

        # 根据箱体类型及突破支撑位/阻力位识别买卖信号
        self.hsa['Pre_Sign']=0

        # 交易策略选取三角形箱体
        
        if Strategic==1:
            # 正收敛三角形0：
            # 突破箱体最近峰值(阻力位)：买入
            self.hsa['Pre_Sign']=self.hsa['Pre_Sign'].mask((self.hsa['Normal Triangle']==0)&(self.hsa['Close_adjusted']>self.hsa['Peak_nearest_shift']),1)
            # 突破箱体最近谷底(支撑位)：卖出
            self.hsa['Pre_Sign']=self.hsa['Pre_Sign'].mask((self.hsa['Normal Triangle']==0)&(self.hsa['Close_adjusted']<self.hsa['Floor_nearest_shift']),-1)
        elif Strategic==2:
            # 向上收敛三角形-1：
            # 突破箱体最近低谷(支撑位)：卖出  （能量耗尽）
            self.hsa['Pre_Sign']=self.hsa['Pre_Sign'].mask((self.hsa['Normal Triangle']==-1)&(self.hsa['Close_adjusted']>self.hsa['Floor_nearest_shift']),-1)
            # 向下收敛三角形1：
            # 突破箱体最近峰值(阻力位)：买入   （能量积累）
            self.hsa['Pre_Sign']=self.hsa['Pre_Sign'].mask((self.hsa['Normal Triangle']==1)&(self.hsa['Close_adjusted']>self.hsa['Peak_nearest_shift']),1)
        elif Strategic==3:
            # 正收敛三角形0：
            # 突破箱体最近峰值(阻力位)：买入
            self.hsa['Pre_Sign']=self.hsa['Pre_Sign'].mask((self.hsa['Normal Triangle']==0)&(self.hsa['Close_adjusted']>self.hsa['Peak_nearest_shift']),1)
            # 突破箱体最近谷底(支撑位)：卖出
            self.hsa['Pre_Sign']=self.hsa['Pre_Sign'].mask((self.hsa['Normal Triangle']==0)&(self.hsa['Close_adjusted']<self.hsa['Floor_nearest_shift']),-1)
            # 向上收敛三角形-1：
            # 突破箱体最近低谷(支撑位)：卖出  （能量耗尽）
            self.hsa['Pre_Sign']=self.hsa['Pre_Sign'].mask((self.hsa['Normal Triangle']==-1)&(self.hsa['Close_adjusted']>self.hsa['Floor_nearest_shift']),-1)
            # 向下收敛三角形1：
            # 突破箱体最近峰值(阻力位)：买入   （能量积累）
            self.hsa['Pre_Sign']=self.hsa['Pre_Sign'].mask((self.hsa['Normal Triangle']==1)&(self.hsa['Close_adjusted']>self.hsa['Peak_nearest_shift']),1)
        elif Strategic==4:
            # 正收敛三角形0：
            # 突破箱体最近峰值(阻力位)：买入
            self.hsa['Pre_Sign']=self.hsa['Pre_Sign'].mask((self.hsa['Normal Triangle']==0)&(self.hsa['Close_adjusted']>self.hsa['Peak_nearest_shift']),1)
            # 突破箱体最近谷底(支撑位)：卖出
            self.hsa['Pre_Sign']=self.hsa['Pre_Sign'].mask((self.hsa['Normal Triangle']==0)&(self.hsa['Close_adjusted']<self.hsa['Floor_nearest_shift']),-1)
            # 向上发散三角形：
            # 直接卖出，能量释放过快，止盈
            self.hsa['Pre_Sign']=self.hsa['Pre_Sign'].mask((self.hsa['Special Triangle']==-2),-1)
            # 向下发散三角形：
            # 直接卖出，能量释放过快，止损
            self.hsa['Pre_Sign']=self.hsa['Pre_Sign'].mask((self.hsa['Special Triangle']==2),-1)
        elif Strategic==5:
            # 正收敛三角形0：
            # 突破箱体最近峰值(阻力位)：买入
            self.hsa['Pre_Sign']=self.hsa['Pre_Sign'].mask((self.hsa['Normal Triangle']==0)&(self.hsa['Close_adjusted']>self.hsa['Peak_nearest_shift']),1)
            # 突破箱体最近谷底(支撑位)：卖出
            self.hsa['Pre_Sign']=self.hsa['Pre_Sign'].mask((self.hsa['Normal Triangle']==0)&(self.hsa['Close_adjusted']<self.hsa['Floor_nearest_shift']),-1)
            # 向上收敛三角形-1：
            # 突破箱体最近低谷(支撑位)：卖出  （能量耗尽）
            self.hsa['Pre_Sign']=self.hsa['Pre_Sign'].mask((self.hsa['Normal Triangle']==-1)&(self.hsa['Close_adjusted']>self.hsa['Floor_nearest_shift']),-1)
            # 向下收敛三角形1：
            # 突破箱体最近峰值(阻力位)：买入   （能量积累）
            self.hsa['Pre_Sign']=self.hsa['Pre_Sign'].mask((self.hsa['Normal Triangle']==1)&(self.hsa['Close_adjusted']>self.hsa['Peak_nearest_shift']),1)
            # 向上发散三角形：
            # 直接卖出，能量释放过快，止盈
            self.hsa['Pre_Sign']=self.hsa['Pre_Sign'].mask((self.hsa['Special Triangle']==-2),-1)
            # 向下发散三角形：
            # 直接卖出，能量释放过快，止损
            self.hsa['Pre_Sign']=self.hsa['Pre_Sign'].mask((self.hsa['Special Triangle']==2),-1)
        elif Strategic==6:
            # 向上收敛三角形-1：
            # 突破箱体最近低谷(支撑位)：卖出  （能量耗尽）
            self.hsa['Pre_Sign']=self.hsa['Pre_Sign'].mask((self.hsa['Normal Triangle']==-1)&(self.hsa['Close_adjusted']>self.hsa['Floor_nearest_shift']),-1)
            # 向下收敛三角形1：
            # 突破箱体最近峰值(阻力位)：买入   （能量积累）
            self.hsa['Pre_Sign']=self.hsa['Pre_Sign'].mask((self.hsa['Normal Triangle']==1)&(self.hsa['Close_adjusted']>self.hsa['Peak_nearest_shift']),1)
            # 向上发散三角形：
            # 直接卖出，能量释放过快，止盈
            self.hsa['Pre_Sign']=self.hsa['Pre_Sign'].mask((self.hsa['Special Triangle']==-2),-1)
            # 向下发散三角形：
            # 直接卖出，能量释放过快，止损
            self.hsa['Pre_Sign']=self.hsa['Pre_Sign'].mask((self.hsa['Special Triangle']==2),-1)
        
        return self.hsa
    
    def Trade(self):
        # 下一期开盘才能购买——故signal滞后一期
        self.hsa['Sign']=self.hsa.groupby('Code')['Pre_Sign'].apply(lambda x: x.shift(1))
        self.hsa=self.hsa.fillna(0)

        # 用复权后的数据计算指标值，用不复权的数据计算持仓价值，然后转换到复权后的数据(手数允许带小数)上进行交易.

        # 历史股票交易情况表
        if self.path[-7:-4]=='SBF':
            Stock_table=self.hsa.copy()[['Code','Name','Date','Open','Close','Open_adjusted','Close_adjusted','High_adjusted','Low_adjusted','Peak_Price','Floor_Price','Pre_Sign','Sign','Delisting']]
        else:
            Stock_table=self.hsa.copy()[['Code','Name','Date','Open','Close','Open_adjusted','Close_adjusted','High_adjusted','Low_adjusted','Peak_Price','Floor_Price','Pre_Sign','Sign']]
            
        self.Date=Stock_table.Date.unique() # Date--索引
        self.Date.sort()
        Stock_table=Stock_table.set_index(['Date'])
        Stock_table['Hold']=0
        Stock_table['Hold_adjusted']=0
        Stock_table['Cost']=0
        Stock_table['Present price']=Stock_table['Close_adjusted']  # 后复权数据
        Stock_table['Return rate']=0
        Stock_table['Profit']=0
        Stock_table['Buy Account']=0
        Stock_table['Sell Account']=0


        # **********最后一天全部卖掉*******
        # Stock_table['Sign']=Stock_table['Sign'].mask((Stock_table.index==self.Date[-1]),-1)
        # 若为Survival Bias Free数据，则对退市的股票进行卖出操作
        if self.path[-7:-4]=='SBF':
            def SBF_sell(x):
                if x.index[-1]<=self.Date[-1]:
                    x.iloc[-1]=-1
                return x
            Stock_table.loc[Stock_table['Delisting']==1,'Sign']=Stock_table.loc[Stock_table['Delisting']==1,].groupby('Code')[['Sign']].apply(SBF_sell)

        # 将买和卖间元素进行标记
        Stock_table['Sign']=Stock_table.groupby('Code')['Sign'].transform(Mark_Sign)

        # 用于画图的买卖信号生成：
        def Point_sign(x):
            x2=x.to_list()
            for i in range(1,len(x2),1):
                if x[i-1]==x[i]:
                    x2[i]=0
            return x2
        Stock_table['buysell_sign']=Stock_table.groupby('Code')['Sign'].transform(Point_sign)

        # 买入股票均持股1手
        Stock_table['Hold']=Stock_table['Hold'].mask(Stock_table['Sign']==1,100)

        # 买入股票计算成本价
        Stock_table['Sign_1']=Stock_table.groupby('Code')['Sign'].apply(lambda x: x.shift(1))
        Stock_table=Stock_table.fillna(0)
        Stock_table['Hold_adjusted']=Stock_table['Hold_adjusted'].mask((Stock_table['Sign']==1)&(Stock_table['Sign_1']!=1),(Stock_table['Open']*100+Stock_table['Open'].apply(buycost)+Stock_table['Open'].apply(sellcost))/Stock_table['Open_adjusted'])
        Stock_table['Hold_adjusted']=Stock_table['Hold_adjusted'].mask((Stock_table['Sign']==1)&(Stock_table['Sign_1']==1),1)
        Stock_table['Hold_adjusted']=Stock_table['Hold_adjusted'].mask((Stock_table['Sign']==-1)&(Stock_table['Sign_1']==1),1)
        # 将买和卖间持股标记上
        Stock_table['Hold_adjusted']=Stock_table.groupby('Code')['Hold_adjusted'].transform(Mark)

        Stock_table['Cost']=Stock_table['Cost'].mask((Stock_table['Sign']==1)&(Stock_table['Sign_1']!=1),Stock_table['Open_adjusted'])
        Stock_table['Cost']=Stock_table['Cost'].mask((Stock_table['Sign']==1)&(Stock_table['Sign_1']==1),1)
        Stock_table['Cost']=Stock_table['Cost'].mask((Stock_table['Sign']==-1)&(Stock_table['Sign_1']==1),1)
        # 将买和卖间成本标记上
        Stock_table['Cost']=Stock_table.groupby('Code')['Cost'].transform(Mark)

        # Return rate of each time and each stock
        Stock_table.loc[Stock_table['Cost']!=0,'Return rate']=(Stock_table.loc[Stock_table['Cost']!=0,'Present price']-Stock_table.loc[Stock_table['Cost']!=0,'Cost'])/Stock_table.loc[Stock_table['Cost']!=0,'Cost']

        # Profit of each sell
        Stock_table['Profit']=Stock_table['Profit'].mask(Stock_table['Sign']==-1,(Stock_table['Open_adjusted']-Stock_table['Cost'])*Stock_table['Hold_adjusted'])

        # Buy Amount + Buy cost + Sell cost
        Stock_table['Buy Account']=Stock_table['Buy Account'].mask((Stock_table['Sign']==1)&(Stock_table['Sign_1']!=1),Stock_table['Cost']*Stock_table['Hold_adjusted'])

        # Sell Amount 
        Stock_table['Sell Account']=Stock_table['Sell Account'].mask(Stock_table['Sign']==-1,Stock_table['Open_adjusted']*Stock_table['Hold_adjusted'])

        # 卖出当天不再持有
        Stock_table['Hold_adjusted']=Stock_table['Hold_adjusted'].mask(Stock_table['Sign']==-1,0)
        self.Stock_table=Stock_table

        return self.Stock_table

    def TC(self):
        ######################### TC ########################
        # 构建历史资金变动表
        Fund_table=pd.DataFrame({'Date':self.Date})
        Fund_table['Account']=self.ora  # 权益
        Fund_table['Liquidity']=self.ora  # 流动资金
        Fund_table['Profit']=0  # 盈亏
        Fund_table['Return rate']=0  # 收益率
        Fund_table['NV']=1  # 净值
        Fund_table['Buy Account']=0  # 买入花费
        Fund_table['Sell Account']=0  # 卖出总额
        Fund_table=Fund_table.set_index('Date')
        # Profit
        Fund_table['Profit']=self.Stock_table.groupby('Date')['Profit'].apply(sum)

        # Buy and Sell Account
        Fund_table['Buy Account']=self.Stock_table.groupby('Date')['Buy Account'].apply(sum)
        Fund_table['Sell Account']=self.Stock_table.groupby('Date')['Sell Account'].apply(sum)

        # Liquidity
        Fund_table['Liquidity']=Fund_table['Liquidity']-Fund_table['Buy Account'].cumsum()+Fund_table['Sell Account'].cumsum()

        # Account
        Fund_table['Account']=Fund_table['Liquidity']+self.Stock_table.groupby('Date').apply(lambda x: (x['Hold_adjusted']*x['Present price']).sum())

        # Return rate
        Fund_table['Return rate']=(Fund_table['Account']-self.ora)/self.ora

        # Net value
        Fund_table['NV']=Fund_table['Account']/self.ora

        self.Fund_table=Fund_table

        return self.Fund_table

    def Table(self):
        ########################### Table #################################
        # 交易次数
        NT=int(self.Stock_table.loc[self.Stock_table['Profit']!=0].shape[0])
        # 胜率
        WR=self.Stock_table.loc[self.Stock_table['Profit']>0].shape[0]/NT
        # 盈亏比
        PLR=-self.Stock_table.loc[self.Stock_table['Profit']>0,'Profit'].mean()/self.Stock_table.loc[self.Stock_table['Profit']<0,'Profit'].mean()
        # 实战率
        # FR=(WR+PLR)/2
        # 250天交易日，3%的无风险收益率
        rf=0.03
        day=250
        # 年化收益率
        IRR=pow(self.Fund_table.loc[self.Date[-1],'NV'],250/len(self.Date))-1
        # 夏普比率
        exceedRR=(self.Fund_table['Account']-self.Fund_table['Account'].shift(1))/self.Fund_table['Account'].shift(1)-rf/day
        SR=np.sqrt(day)*exceedRR.mean()/exceedRR.std()
        # 最大回撤
        MD=((self.Fund_table['NV'].cummax()-self.Fund_table['NV'])/self.Fund_table['NV'].cummax()).max()

        # 策略技术性指标
        report=pd.DataFrame(np.zeros((5,1),dtype=float),index=['交易次数','胜率','盈亏比','夏普比率','最大回撤'],columns=['技术性指标'])
        report.loc['交易次数']=NT
        report.loc['胜率']=WR
        report.loc['盈亏比']=PLR
        # report.loc['实战率']=FR
        report.loc['年化收益率']=IRR
        report.loc['夏普比率']=SR
        report.loc['最大回撤']=MD
        pd.set_option('display.float_format',  '{:,.3f}'.format) # 显示小数点后三位
        report=report.T
        format_dict = {'交易次数': '{0:.0f}', 
                        '胜率': '{0:.2%}',
                        '盈亏比':'{0:.3f}',
                        # '实战率':'{0:.2%}',
                        '年化收益率':'{0:.2%}',
                        '夏普比率':'{0:.3f}',
                        '最大回撤':'{0:.2%}',
                        }
        return report.style.format(format_dict)


    def TC_Graph(self):
        plt.figure(figsize=(15,70))
        plt.subplot(7,1,1)
        plt.plot(self.Fund_table['Account'],'-b')
        plt.title('Account')
        plt.subplot(7,1,2)
        plt.plot(self.Fund_table['Liquidity'],'-b')
        plt.title('Liquidity')
        ax1=plt.subplot(7,1,3)
        self.Fund_table['Profit'].plot(kind="bar",ax=ax1)
        ax1.set_xticklabels([x.strftime("%Y-%m-%d") for x in self.Fund_table.index], rotation=30)
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(int(len(self.Date)/20)))
        plt.title('Profit')
        plt.subplot(7,1,4)
        plt.plot(self.Fund_table['Return rate'],'-b')
        plt.title('Return rate')
        plt.subplot(7,1,5)
        plt.plot(self.Fund_table['NV'],'-b')
        plt.title('Net Value')
        ax2=plt.subplot(7,1,6)
        self.Fund_table['Buy Account'].plot(kind="bar",ax=ax2)
        ax2.set_xticklabels([x.strftime("%Y-%m-%d") for x in self.Fund_table.index], rotation=30)
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(int(len(self.Date)/20)))
        plt.title('Buy Account')
        ax3=plt.subplot(7,1,7)
        self.Fund_table['Sell Account'].plot(kind="bar",ax=ax3)
        ax3.set_xticklabels([x.strftime("%Y-%m-%d") for x in self.Fund_table.index], rotation=30)
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(int(len(self.Date)/20)))
        plt.title('Sell Account')

    def Single_Fig(self):
        if self.single_stock[0]:
            stock=self.Stock_table[['Close_adjusted','Open_adjusted','Low_adjusted','High_adjusted','Peak_Price','Floor_Price','buysell_sign']]
            stock.columns=['Close','Open','Low','High','Peak_Price','Floor_Price','Sign']
            buy_sign=stock[['Open','Sign']].copy()
            buy_sign.loc[buy_sign['Sign']!=1,'Open']=None
            sell_sign=stock[['Open','Sign']].copy()
            sell_sign.loc[sell_sign['Sign']!=-1,'Open']=None
            buy_sign=buy_sign['Open']
            sell_sign=sell_sign['Open']
            peak_sign=stock['Peak_Price']
            peak_sign.loc[peak_sign==0]=None
            floor_sign=stock['Floor_Price']
            floor_sign.loc[floor_sign==0]=None

            # 设置mplfinance的蜡烛颜色，up为阳线颜色，down为阴线颜色
            my_color = mpf.make_marketcolors(up='r',
                                            down='g',
                                            edge='inherit',
                                            wick='inherit',
                                            volume='inherit')
            # 设置图表的背景色
            my_style = mpf.make_mpf_style(marketcolors=my_color,
                                        figcolor='(0.82, 0.83, 0.85)',
                                        gridcolor='(0.82, 0.83, 0.85)')
            add_plot=[]

            # 买卖信号 上三角买入，下三角卖出
            add_plot=add_plot+[
                mpf.make_addplot(buy_sign,scatter=True,marker="^",markersize=80,color="b",secondary_y=False),
                mpf.make_addplot(sell_sign,scatter=True,marker="v",markersize=80,color="k",secondary_y=False)
                ]
            # 红色峰值 绿色谷底
            add_plot=add_plot+[
                mpf.make_addplot(peak_sign,scatter=True,marker="o",markersize=80,color="r",secondary_y=False),
                mpf.make_addplot(floor_sign,scatter=True,marker="o",markersize=80,color="g",secondary_y=False)
            ]


            mpf.plot(stock, type='candle',
                            addplot=add_plot,
                            volume=False,
                            figscale=1.5,
                            style=my_style,
                            figratio=(8,5),
                            main_panel=0, 
                            # volume_panel=1,
                            )

            plt.show()  # 显示
    

# 交易费用
def buycost(price):
    h=price*100
    if h*0.0003>=5:
        yj=h*0.0003 
    else:
        yj=5
    return h*0.0000687+h*0.00002+yj

def sellcost(price):
    h=price*100
    if h*0.0003>=5:
        yj=h*0.0003 
    else:
        yj=5
    return h*0.0000687+h*0.00002+yj+h*0.001

def Mark_Sign(x):
    x2=x.to_list()
    for i in range(1,len(x2),1):
        if x2[i-1]==1 and x2[i]!=-1:
            x2[i]=1
        elif x2[i-1]!=1 and x2[i]==-1:
            x2[i]=0
    return x2

def Mark(x):
    x2=x.to_list()
    for i in range(1,len(x2),1):
        if x[i]==1:
            x2[i]=x2[i-1]
    return x2

def Fill(x,Shift):
    if Shift:
        x=x.shift(1)
    else:
        x=x
    x2=x.to_list()
    for i in range(1,len(x2),1):
        if x2[i]==0:
            x2[i]=x2[i-1]
    return x2
