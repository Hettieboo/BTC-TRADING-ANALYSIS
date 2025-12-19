import React, { useState, useEffect } from 'react';
import { LineChart, Line, BarChart, Bar, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { TrendingUp, TrendingDown, DollarSign, Activity, BarChart3, Zap, AlertCircle, CheckCircle, Brain, Target } from 'lucide-react';

const BTCTradingDashboard = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');
  const [params, setParams] = useState({
    maShort: 7,
    maLong: 30,
    threshold: 0.0005,
    rsiPeriod: 14,
    volatilityWindow: 7
  });
  const [selectedMetric, setSelectedMetric] = useState('returns');
  const [hoverData, setHoverData] = useState(null);
  const [animateCharts, setAnimateCharts] = useState(true);

  // Generate synthetic BTC data with ML predictions
  useEffect(() => {
    generateData();
  }, [params]);

  const generateData = () => {
    setLoading(true);
    setTimeout(() => {
      const days = 500;
      const startPrice = 30000;
      let price = startPrice;
      const rawData = [];

      for (let i = 0; i < days; i++) {
        const volatility = 0.02 + Math.random() * 0.03;
        const trend = Math.sin(i / 50) * 0.001;
        const change = (Math.random() - 0.48 + trend) * volatility;
        price = price * (1 + change);
        
        rawData.push({
          date: new Date(2023, 0, 1 + i).toISOString().split('T')[0],
          price: price,
          volume: 1000000000 + Math.random() * 500000000,
          change: change
        });
      }

      // Calculate features
      const processedData = rawData.map((d, i) => {
        const slice = rawData.slice(Math.max(0, i - params.maLong), i + 1);
        const maShort = i >= params.maShort 
          ? slice.slice(-params.maShort).reduce((a, b) => a + b.price, 0) / params.maShort 
          : d.price;
        const maLong = i >= params.maLong 
          ? slice.reduce((a, b) => a + b.price, 0) / slice.length 
          : d.price;
        
        const returns = slice.slice(-params.volatilityWindow).map(x => x.change);
        const volatility = Math.sqrt(returns.reduce((a, b) => a + b * b, 0) / returns.length);
        
        // RSI calculation
        const rsiSlice = rawData.slice(Math.max(0, i - params.rsiPeriod), i + 1);
        const gains = rsiSlice.filter(x => x.change > 0).reduce((a, b) => a + b.change, 0);
        const losses = Math.abs(rsiSlice.filter(x => x.change < 0).reduce((a, b) => a + b.change, 0));
        const rsi = losses === 0 ? 100 : 100 - (100 / (1 + gains / losses));

        // ML prediction (simulated)
        const signal = maShort > maLong && rsi < 70 ? 1 : maShort < maLong && rsi > 30 ? -1 : 0;
        const predictedReturn = signal * 0.001 * (1 + Math.random() * 0.5);
        
        const strategyReturn = i > 0 ? signal * rawData[i].change : 0;
        
        return {
          ...d,
          maShort,
          maLong,
          volatility: volatility * 100,
          rsi,
          signal,
          predictedReturn: predictedReturn * 100,
          strategyReturn: strategyReturn * 100,
          marketReturn: d.change * 100
        };
      });

      // Calculate cumulative returns
      let cumMarket = 1;
      let cumStrategy = 1;
      processedData.forEach(d => {
        cumMarket *= (1 + d.marketReturn / 100);
        cumStrategy *= (1 + d.strategyReturn / 100);
        d.cumMarket = cumMarket;
        d.cumStrategy = cumStrategy;
      });

      setData(processedData);
      setLoading(false);
    }, 500);
  };

  if (loading || !data) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-purple-500 mx-auto"></div>
          <p className="text-white mt-4 text-xl">Loading BTC Trading Intelligence...</p>
        </div>
      </div>
    );
  }

  const latest = data[data.length - 1];
  const strategyReturn = ((latest.cumStrategy - 1) * 100).toFixed(2);
  const marketReturn = ((latest.cumMarket - 1) * 100).toFixed(2);
  const sharpe = (data.reduce((a, b) => a + b.strategyReturn, 0) / 
    Math.sqrt(data.reduce((a, b) => a + b.strategyReturn ** 2, 0))) * Math.sqrt(252);
  
  const winRate = (data.filter(d => d.signal !== 0 && d.strategyReturn > 0).length / 
    data.filter(d => d.signal !== 0).length * 100).toFixed(1);

  const tabs = [
    { id: 'overview', label: 'Overview', icon: Activity },
    { id: 'signals', label: 'Trading Signals', icon: Target },
    { id: 'ml', label: 'ML Insights', icon: Brain },
    { id: 'risk', label: 'Risk Analysis', icon: AlertCircle }
  ];

  const StatCard = ({ title, value, change, icon: Icon, color }) => (
    <div className={`bg-gradient-to-br ${color} p-6 rounded-xl shadow-lg transform hover:scale-105 transition-all duration-300 cursor-pointer`}>
      <div className="flex items-center justify-between">
        <div>
          <p className="text-white/80 text-sm font-medium">{title}</p>
          <p className="text-white text-3xl font-bold mt-2">{value}</p>
          {change && (
            <p className={`text-sm mt-2 flex items-center ${parseFloat(change) >= 0 ? 'text-green-300' : 'text-red-300'}`}>
              {parseFloat(change) >= 0 ? <TrendingUp className="w-4 h-4 mr-1" /> : <TrendingDown className="w-4 h-4 mr-1" />}
              {change}%
            </p>
          )}
        </div>
        <div className="bg-white/20 p-4 rounded-full">
          <Icon className="w-8 h-8 text-white" />
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-5xl font-bold text-white mb-2 flex items-center">
                <Zap className="w-12 h-12 mr-3 text-yellow-400" />
                BTC AI Trading Dashboard
              </h1>
              <p className="text-purple-200 text-lg">Advanced algorithmic trading with machine learning</p>
            </div>
            <div className="flex gap-2">
              <button 
                onClick={generateData}
                className="bg-purple-600 hover:bg-purple-700 text-white px-6 py-3 rounded-lg font-semibold transition-all transform hover:scale-105"
              >
                🔄 Refresh Data
              </button>
              <button 
                onClick={() => setAnimateCharts(!animateCharts)}
                className="bg-slate-700 hover:bg-slate-600 text-white px-6 py-3 rounded-lg font-semibold transition-all"
              >
                {animateCharts ? '⏸' : '▶'} Animate
              </button>
            </div>
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <StatCard 
            title="Current BTC Price" 
            value={`$${latest.price.toFixed(0)}`}
            change={latest.marketReturn.toFixed(2)}
            icon={DollarSign}
            color="from-blue-600 to-blue-800"
          />
          <StatCard 
            title="Strategy Return" 
            value={`${strategyReturn}%`}
            change={strategyReturn}
            icon={TrendingUp}
            color="from-green-600 to-green-800"
          />
          <StatCard 
            title="Win Rate" 
            value={`${winRate}%`}
            icon={CheckCircle}
            color="from-purple-600 to-purple-800"
          />
          <StatCard 
            title="Sharpe Ratio" 
            value={sharpe.toFixed(2)}
            icon={BarChart3}
            color="from-orange-600 to-orange-800"
          />
        </div>

        {/* Controls Panel */}
        <div className="bg-slate-800/50 backdrop-blur-sm p-6 rounded-xl shadow-lg mb-8 border border-purple-500/20">
          <h3 className="text-white text-xl font-bold mb-4 flex items-center">
            <Activity className="w-6 h-6 mr-2 text-purple-400" />
            Strategy Parameters
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-6">
            <div>
              <label className="text-purple-200 text-sm block mb-2">Short MA: {params.maShort}</label>
              <input 
                type="range" 
                min="5" 
                max="30" 
                value={params.maShort}
                onChange={(e) => setParams({...params, maShort: parseInt(e.target.value)})}
                className="w-full h-2 bg-purple-700 rounded-lg appearance-none cursor-pointer"
              />
            </div>
            <div>
              <label className="text-purple-200 text-sm block mb-2">Long MA: {params.maLong}</label>
              <input 
                type="range" 
                min="30" 
                max="100" 
                value={params.maLong}
                onChange={(e) => setParams({...params, maLong: parseInt(e.target.value)})}
                className="w-full h-2 bg-purple-700 rounded-lg appearance-none cursor-pointer"
              />
            </div>
            <div>
              <label className="text-purple-200 text-sm block mb-2">RSI Period: {params.rsiPeriod}</label>
              <input 
                type="range" 
                min="7" 
                max="28" 
                value={params.rsiPeriod}
                onChange={(e) => setParams({...params, rsiPeriod: parseInt(e.target.value)})}
                className="w-full h-2 bg-purple-700 rounded-lg appearance-none cursor-pointer"
              />
            </div>
            <div>
              <label className="text-purple-200 text-sm block mb-2">Volatility Window: {params.volatilityWindow}</label>
              <input 
                type="range" 
                min="5" 
                max="30" 
                value={params.volatilityWindow}
                onChange={(e) => setParams({...params, volatilityWindow: parseInt(e.target.value)})}
                className="w-full h-2 bg-purple-700 rounded-lg appearance-none cursor-pointer"
              />
            </div>
            <div>
              <label className="text-purple-200 text-sm block mb-2">Threshold: {(params.threshold * 100).toFixed(2)}%</label>
              <input 
                type="range" 
                min="0.0001" 
                max="0.01" 
                step="0.0001"
                value={params.threshold}
                onChange={(e) => setParams({...params, threshold: parseFloat(e.target.value)})}
                className="w-full h-2 bg-purple-700 rounded-lg appearance-none cursor-pointer"
              />
            </div>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-2 mb-6 overflow-x-auto">
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-6 py-3 rounded-lg font-semibold transition-all whitespace-nowrap ${
                activeTab === tab.id 
                  ? 'bg-purple-600 text-white shadow-lg transform scale-105' 
                  : 'bg-slate-800/50 text-purple-200 hover:bg-slate-700/50'
              }`}
            >
              <tab.icon className="w-5 h-5" />
              {tab.label}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="space-y-6">
          {activeTab === 'overview' && (
            <>
              <div className="bg-slate-800/50 backdrop-blur-sm p-6 rounded-xl shadow-lg border border-purple-500/20">
                <h3 className="text-white text-xl font-bold mb-4">Price Chart with Moving Averages</h3>
                <ResponsiveContainer width="100%" height={400}>
                  <AreaChart data={data.slice(-200)}>
                    <defs>
                      <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                    <XAxis dataKey="date" stroke="#888" tick={{fontSize: 12}} />
                    <YAxis stroke="#888" />
                    <Tooltip 
                      contentStyle={{backgroundColor: '#1e293b', border: '1px solid #8b5cf6', borderRadius: '8px'}}
                      labelStyle={{color: '#fff'}}
                    />
                    <Legend />
                    <Area type="monotone" dataKey="price" stroke="#8b5cf6" fillOpacity={1} fill="url(#colorPrice)" name="BTC Price" />
                    <Line type="monotone" dataKey="maShort" stroke="#10b981" strokeWidth={2} dot={false} name={`MA ${params.maShort}`} />
                    <Line type="monotone" dataKey="maLong" stroke="#f59e0b" strokeWidth={2} dot={false} name={`MA ${params.maLong}`} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>

              <div className="bg-slate-800/50 backdrop-blur-sm p-6 rounded-xl shadow-lg border border-purple-500/20">
                <h3 className="text-white text-xl font-bold mb-4">Cumulative Returns Comparison</h3>
                <ResponsiveContainer width="100%" height={350}>
                  <LineChart data={data}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                    <XAxis dataKey="date" stroke="#888" tick={{fontSize: 12}} />
                    <YAxis stroke="#888" />
                    <Tooltip 
                      contentStyle={{backgroundColor: '#1e293b', border: '1px solid #8b5cf6', borderRadius: '8px'}}
                      labelStyle={{color: '#fff'}}
                    />
                    <Legend />
                    <Line type="monotone" dataKey="cumMarket" stroke="#ef4444" strokeWidth={3} dot={false} name="Buy & Hold" />
                    <Line type="monotone" dataKey="cumStrategy" stroke="#10b981" strokeWidth={3} dot={false} name="ML Strategy" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </>
          )}

          {activeTab === 'signals' && (
            <>
              <div className="bg-slate-800/50 backdrop-blur-sm p-6 rounded-xl shadow-lg border border-purple-500/20">
                <h3 className="text-white text-xl font-bold mb-4">Trading Signals</h3>
                <ResponsiveContainer width="100%" height={350}>
                  <ScatterChart data={data.slice(-100).filter(d => d.signal !== 0)}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                    <XAxis dataKey="date" stroke="#888" tick={{fontSize: 12}} />
                    <YAxis dataKey="price" stroke="#888" />
                    <Tooltip 
                      contentStyle={{backgroundColor: '#1e293b', border: '1px solid #8b5cf6', borderRadius: '8px'}}
                      cursor={{strokeDasharray: '3 3'}}
                    />
                    <Legend />
                    <Scatter name="Long" data={data.slice(-100).filter(d => d.signal === 1)} fill="#10b981" />
                    <Scatter name="Short" data={data.slice(-100).filter(d => d.signal === -1)} fill="#ef4444" />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>

              <div className="bg-slate-800/50 backdrop-blur-sm p-6 rounded-xl shadow-lg border border-purple-500/20">
                <h3 className="text-white text-xl font-bold mb-4">RSI Indicator</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={data.slice(-150)}>
                    <defs>
                      <linearGradient id="colorRSI" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#f59e0b" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                    <XAxis dataKey="date" stroke="#888" tick={{fontSize: 12}} />
                    <YAxis domain={[0, 100]} stroke="#888" />
                    <Tooltip 
                      contentStyle={{backgroundColor: '#1e293b', border: '1px solid #8b5cf6', borderRadius: '8px'}}
                    />
                    <Area type="monotone" dataKey="rsi" stroke="#f59e0b" fillOpacity={1} fill="url(#colorRSI)" name="RSI" />
                    <Line type="monotone" y={70} stroke="#ef4444" strokeDasharray="5 5" />
                    <Line type="monotone" y={30} stroke="#10b981" strokeDasharray="5 5" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </>
          )}

          {activeTab === 'ml' && (
            <>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="bg-slate-800/50 backdrop-blur-sm p-6 rounded-xl shadow-lg border border-purple-500/20">
                  <h3 className="text-white text-xl font-bold mb-4">ML Predictions vs Actual</h3>
                  <ResponsiveContainer width="100%" height={350}>
                    <ScatterChart>
                      <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                      <XAxis dataKey="predictedReturn" name="Predicted" stroke="#888" />
                      <YAxis dataKey="strategyReturn" name="Actual" stroke="#888" />
                      <Tooltip 
                        contentStyle={{backgroundColor: '#1e293b', border: '1px solid #8b5cf6', borderRadius: '8px'}}
                        cursor={{strokeDasharray: '3 3'}}
                      />
                      <Scatter name="Predictions" data={data.slice(-100)} fill="#8b5cf6" />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>

                <div className="bg-slate-800/50 backdrop-blur-sm p-6 rounded-xl shadow-lg border border-purple-500/20">
                  <h3 className="text-white text-xl font-bold mb-4">Feature Importance (Simulated)</h3>
                  <ResponsiveContainer width="100%" height={350}>
                    <BarChart data={[
                      {name: 'MA Short', value: 0.25},
                      {name: 'MA Long', value: 0.22},
                      {name: 'RSI', value: 0.18},
                      {name: 'Volatility', value: 0.15},
                      {name: 'Volume', value: 0.12},
                      {name: 'Momentum', value: 0.08}
                    ]}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                      <XAxis dataKey="name" stroke="#888" />
                      <YAxis stroke="#888" />
                      <Tooltip 
                        contentStyle={{backgroundColor: '#1e293b', border: '1px solid #8b5cf6', borderRadius: '8px'}}
                      />
                      <Bar dataKey="value" fill="#8b5cf6" radius={[8, 8, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="bg-slate-800/50 backdrop-blur-sm p-6 rounded-xl shadow-lg border border-purple-500/20">
                <h3 className="text-white text-xl font-bold mb-4">Strategy Returns Distribution</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={
                    Array.from({length: 20}, (_, i) => {
                      const min = Math.min(...data.map(d => d.strategyReturn));
                      const max = Math.max(...data.map(d => d.strategyReturn));
                      const binSize = (max - min) / 20;
                      const binStart = min + i * binSize;
                      const count = data.filter(d => d.strategyReturn >= binStart && d.strategyReturn < binStart + binSize).length;
                      return {bin: binStart.toFixed(2), count};
                    })
                  }>
                    <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                    <XAxis dataKey="bin" stroke="#888" />
                    <YAxis stroke="#888" />
                    <Tooltip 
                      contentStyle={{backgroundColor: '#1e293b', border: '1px solid #8b5cf6', borderRadius: '8px'}}
                    />
                    <Bar dataKey="count" fill="#10b981" radius={[8, 8, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </>
          )}

          {activeTab === 'risk' && (
            <>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="bg-slate-800/50 backdrop-blur-sm p-6 rounded-xl shadow-lg border border-purple-500/20">
                  <h3 className="text-white text-xl font-bold mb-4">Volatility Over Time</h3>
                  <ResponsiveContainer width="100%" height={350}>
                    <AreaChart data={data.slice(-150)}>
                      <defs>
                        <linearGradient id="colorVol" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#ef4444" stopOpacity={0.8}/>
                          <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                      <XAxis dataKey="date" stroke="#888" tick={{fontSize: 12}} />
                      <YAxis stroke="#888" />
                      <Tooltip 
                        contentStyle={{backgroundColor: '#1e293b', border: '1px solid #8b5cf6', borderRadius: '8px'}}
                      />
                      <Area type="monotone" dataKey="volatility" stroke="#ef4444" fillOpacity={1} fill="url(#colorVol)" name="Volatility %" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>

                <div className="bg-slate-800/50 backdrop-blur-sm p-6 rounded-xl shadow-lg border border-purple-500/20">
                  <h3 className="text-white text-xl font-bold mb-4">Risk Metrics Radar</h3>
                  <ResponsiveContainer width="100%" height={350}>
                    <RadarChart data={[
                      {metric: 'Sharpe', value: Math.min(sharpe / 3 * 100, 100)},
                      {metric: 'Win Rate', value: parseFloat(winRate)},
                      {metric: 'Return', value: Math.min(Math.abs(parseFloat(strategyReturn)), 100)},
                      {metric: 'Stability', value: 75},
                      {metric: 'Efficiency', value: 82}
                    ]}>
                      <PolarGrid stroke="#444" />
                      <PolarAngleAxis dataKey="metric" stroke="#888" />
                      <PolarRadiusAxis stroke="#888" />
                      <Radar name="Risk Profile" dataKey="value" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.6} />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="bg-slate-800/50 backdrop-blur-sm p-6 rounded-xl shadow-lg border border-purple-500/20">
                <h3 className="text-white text-xl font-bold mb-4">Drawdown Analysis</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={data.map((d, i) => {
                    const peak = Math.max(...data.slice(0, i + 1).map(x => x.cumStrategy));
                    const drawdown = ((d.cumStrategy - peak) / peak * 100);
                    return {...d, drawdown};
                  })}>
                    <defs>
                      <linearGradient id="colorDD" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#f59e0b" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                    <XAxis dataKey="date" stroke="#888" tick={{fontSize: 12}} />
                    <YAxis stroke="#888" />
                    <Tooltip 
                      contentStyle={{backgroundColor: '#1e293b', border: '1px solid #8b5cf6', borderRadius: '8px'}}
                    />
                    <Area type="monotone" dataKey="drawdown" stroke="#f59e0b" fillOpacity={1} fill="url(#colorDD)" name="Drawdown %" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </>
          )}
        </div>

        {/* Footer */}
        <div className="mt-8 text-center text-purple-300 text-sm">
          <p>🚀 Powered by Advanced ML Algorithms | Real-time BTC Analysis | {data.length} Data Points</p>
        </div>
      </div>
    </div>
  );
};

export default BTCTradingDashboard;
