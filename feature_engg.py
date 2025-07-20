import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime
from typing import Dict, List
from scipy import stats
import matplotlib.pyplot as plt

from fraud_detect import FraudDetectionModule
from risk_asses import RiskAssessmentModule
from predictive_analysis import PredictiveAnalyticsModule

class AdvancedFeatureEngineering:
    
    def __init__(self):
        self.fraud_detector_instance = FraudDetectionModule()

    def engineer_advanced_features(self, transactions_data: List[Dict]) -> pd.DataFrame:
        features_list = []
        all_transactions = transactions_data
        
        wallet_groups = {}
        for tx in transactions_data:
            wallet = tx['userWallet']
            if wallet not in wallet_groups:
                wallet_groups[wallet] = []
            wallet_groups[wallet].append(tx)
        
        coordination_scores = self.fraud_detector_instance.detect_coordinated_behavior(all_transactions)
        
        for wallet, txs in wallet_groups.items():
            features = {'wallet': wallet}
            
            features.update(self._calculate_basic_features(txs))
            
            network_features = self.calculate_network_features(txs, wallet)
            features.update(network_features)
            
            features['behavioral_entropy'] = self.calculate_behavioral_entropy(txs)
            
            ts_features = self.calculate_time_series_features(txs)
            features.update(ts_features)
            
            risk_assessment_module = RiskAssessmentModule()
            portfolio_data = self._extract_portfolio_data(txs)
            features['var_risk'] = risk_assessment_module.calculate_var_risk(portfolio_data)
            features['expected_shortfall'] = risk_assessment_module.calculate_expected_shortfall(portfolio_data)
            
            stress_results = risk_assessment_module.stress_test_portfolio(features)
            features.update({f'stress_{k}': v for k, v in stress_results.items()})
            
            features['circular_transactions'] = self.fraud_detector_instance.detect_circular_transactions(txs)
            features['coordination_score'] = coordination_scores.get(wallet, 0)
            
            predictive_module = PredictiveAnalyticsModule()
            features['default_probability'] = predictive_module.predict_default_probability(features)
            liquidity_pred = predictive_module.predict_liquidity_needs(txs)
            features.update({f'pred_{k}': v for k, v in liquidity_pred.items()})
            
            features.update(self._calculate_statistical_features(txs))
            
            features_list.append(features)
        
        return pd.DataFrame(features_list).set_index('wallet')

    @staticmethod
    def calculate_network_features(transactions: List[Dict], user_wallet: str) -> Dict:
        G = nx.DiGraph()
        
        for tx in transactions:
            wallet = tx.get('userWallet')
            if 'actionData' in tx:
                asset = tx['actionData'].get('assetSymbol', 'unknown')
                pool = tx['actionData'].get('poolId', 'unknown')
                
                G.add_node(wallet, type='wallet')
                G.add_node(asset, type='asset')
                if pool != 'unknown':
                    G.add_node(pool, type='pool')
                    G.add_edge(wallet, pool, weight=1, tx_type=tx['action'])
                    G.add_edge(pool, asset, weight=1, tx_type=tx['action'])
                G.add_edge(wallet, asset, weight=1, tx_type=tx['action'])
        
        network_features = {}
        if G.number_of_nodes() > 0:
            if user_wallet in G:
                network_features['network_degree_centrality'] = nx.degree_centrality(G).get(user_wallet, 0)
                network_features['network_clustering'] = nx.clustering(G.to_undirected()).get(user_wallet, 0)
                network_features['network_betweenness_centrality'] = nx.betweenness_centrality(G).get(user_wallet, 0)
            else:
                network_features['network_degree_centrality'] = 0
                network_features['network_clustering'] = 0
                network_features['network_betweenness_centrality'] = 0

            network_features['connected_components'] = nx.number_connected_components(G.to_undirected())
        else:
            network_features = {k: 0 for k in ['network_degree_centrality', 'network_clustering', 
                                              'network_betweenness_centrality', 'connected_components']}
        
        return network_features
    
    @staticmethod
    def calculate_behavioral_entropy(transactions: List[Dict]) -> float:
        if not transactions:
            return 0
        
        actions = [tx['action'] for tx in transactions]
        action_counts = pd.Series(actions).value_counts()
        probabilities = action_counts / len(actions)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy
    
    @staticmethod
    def calculate_time_series_features(transactions: List[Dict]) -> Dict:
        if len(transactions) < 3:
            return {f'ts_{feature}': 0 for feature in 
                    ['trend', 'seasonality', 'volatility', 'autocorr', 'stationarity']}
        
        sorted_txs = sorted(transactions, key=lambda x: x['timestamp'])
        
        amounts = []
        timestamps = []
        for tx in sorted_txs:
            if 'actionData' in tx:
                amount = float(tx['actionData'].get('amount', 0))
                amounts.append(amount)
                timestamps.append(tx['timestamp'])
        
        if len(amounts) < 3:
            return {f'ts_{feature}': 0 for feature in 
                    ['trend', 'seasonality', 'volatility', 'autocorr', 'stationarity']}
        
        ts = pd.Series(amounts, index=pd.to_datetime(timestamps, unit='s'))
        ts = ts.resample('D').sum().fillna(0)
        
        features = {}
        
        if len(ts) > 1:
            slope, _, _, _, _ = stats.linregress(range(len(ts)), ts.values + 1e-10) 
            features['ts_trend'] = slope
        else:
            features['ts_trend'] = 0
        
        features['ts_volatility'] = ts.std() / (ts.mean() + 1e-10)
        
        if len(ts) > 2:
            features['ts_autocorr'] = ts.autocorr(lag=1) if not ts.empty else 0
        else:
            features['ts_autocorr'] = 0
        
        features['ts_stationarity'] = 1 / (1 + features['ts_volatility'] + abs(features['ts_trend']))
        
        if len(ts) > 7:
            features['ts_seasonality'] = len(set(ts.index.dayofweek)) / 7.0
        else:
            features['ts_seasonality'] = 0
        
        return features
    
    def _calculate_basic_features(self, txs: List[Dict]) -> Dict:
        features = {}
        
        txs_sorted = sorted(txs, key=lambda x: x['timestamp'])
        
        features['total_transactions'] = len(txs)
        features['unique_actions'] = len(set(tx['action'] for tx in txs))
        
        def get_usd_amount(tx):
            try:
                if 'actionData' in tx:
                    amount = float(tx['actionData'].get('amount', 0))
                    price = float(tx['actionData'].get('assetPriceUSD', 0))
                    asset_symbol = tx['actionData'].get('assetSymbol', '')
                    if asset_symbol in ['USDC', 'USDT', 'DAI']:
                        amount_adjusted = amount / 1e6 
                    else:
                        amount_adjusted = amount / 1e18
                    return amount_adjusted * price
                return 0
            except (ValueError, TypeError):
                return 0
        
        actions = ['deposit', 'borrow', 'repay', 'redeemUnderlying', 'liquidationCall']
        for action in actions:
            action_txs = [tx for tx in txs if tx['action'] == action]
            features[f'{action}_count'] = len(action_txs)
            usd_amounts = [get_usd_amount(tx) for tx in action_txs]
            features[f'{action}_total_usd'] = sum(usd_amounts)
            features[f'{action}_avg_usd'] = features[f'{action}_total_usd'] / max(1, features[f'{action}_count'])
        
        total_deposits = features['deposit_total_usd']
        total_borrows = features['borrow_total_usd']
        total_repays = features['repay_total_usd']
        total_liquidations = features['liquidationCall_total_usd']
        
        features['repay_to_borrow_ratio'] = total_repays / max(1, total_borrows)
        features['deposit_to_borrow_ratio'] = total_deposits / max(1, total_borrows)
        features['liquidation_ratio'] = total_liquidations / max(1, total_deposits + total_borrows)
        
        unique_assets = set()
        for tx in txs:
            if 'actionData' in tx:
                if 'assetSymbol' in tx['actionData']:
                    unique_assets.add(tx['actionData']['assetSymbol'])
                if 'poolId' in tx['actionData']:
                    unique_assets.add(tx['actionData']['poolId'])
        features['unique_assets'] = len(unique_assets)
        
        if len(txs_sorted) > 1:
            timestamps = [tx['timestamp'] for tx in txs_sorted]
            time_diffs = np.diff(timestamps)
            features['avg_time_between_txs'] = np.mean(time_diffs)
            features['std_time_between_txs'] = np.std(time_diffs)
            features['activity_span_days'] = (timestamps[-1] - timestamps[0]) / (24 * 3600)
            features['tx_frequency'] = features['total_transactions'] / max(1, features['activity_span_days'])
        else:
            features.update({
                'avg_time_between_txs': 0, 'std_time_between_txs': 0,
                'activity_span_days': 0, 'tx_frequency': 0
            })
        
        return features

    def _calculate_statistical_features(self, txs: List[Dict]) -> Dict:
        features = {}
        
        if not txs:
            return {f'stat_{k}': 0 for k in ['skewness', 'kurtosis', 'gini_coef', 'hurst_exp']}
        
        amounts = []
        for tx in txs:
            if 'actionData' in tx:
                amount = float(tx['actionData'].get('amount', 0))
                if amount > 0:
                    amounts.append(amount)
        
        if not amounts:
            return {f'stat_{k}': 0 for k in ['skewness', 'kurtosis', 'gini_coef', 'hurst_exp']}
        
        amounts = np.array(amounts)
        
        features['stat_skewness'] = stats.skew(amounts)
        features['stat_kurtosis'] = stats.kurtosis(amounts)
        
        if np.sum(amounts) == 0:
            features['stat_gini_coef'] = 0
        else:
            sorted_amounts = np.sort(amounts)
            n = len(amounts)
            cumsum = np.cumsum(sorted_amounts)
            features['stat_gini_coef'] = (n + 1 - 2 * np.sum(cumsum) / (cumsum[-1] + 1e-10)) / n
            
        if len(amounts) > 10:
            try:
                lags = range(2, min(len(amounts)//2, 20))
                tau = [np.sqrt(np.std(np.subtract(amounts[lag:], amounts[:-lag]))) for lag in lags if len(amounts)-lag > 0]
                
                valid_lags = [lags[i] for i, val in enumerate(tau) if val > 0]
                valid_tau = [val for val in tau if val > 0]

                if len(valid_tau) > 1:
                    poly = np.polyfit(np.log(valid_lags), np.log(valid_tau), 1)
                    features['stat_hurst_exp'] = poly[0]
                else:
                    features['stat_hurst_exp'] = 0.5
            except Exception:
                features['stat_hurst_exp'] = 0.5
        else:
            features['stat_hurst_exp'] = 0.5
        
        return features

    def _extract_portfolio_data(self, txs: List[Dict]) -> Dict:
        portfolio_data = {'returns': [], 'positions': {}}
        
        daily_net_value = {}
        for tx in txs:
            if 'actionData' in tx:
                date = datetime.fromtimestamp(tx['timestamp']).date()
                amount_usd = 0
                
                def get_usd_value(tx_data):
                    try:
                        amount = float(tx_data.get('amount', 0))
                        price = float(tx_data.get('assetPriceUSD', 0))
                        asset_symbol = tx_data.get('assetSymbol', '')
                        if asset_symbol in ['USDC', 'USDT', 'DAI']:
                            return (amount / 1e6) * price 
                        else:
                            return (amount / 1e18) * price
                    except (ValueError, TypeError):
                        return 0
                
                amount_usd = get_usd_value(tx['actionData'])

                if tx['action'] == 'deposit':
                    daily_net_value[date] = daily_net_value.get(date, 0) + amount_usd
                elif tx['action'] == 'borrow':
                    daily_net_value[date] = daily_net_value.get(date, 0) - amount_usd
                elif tx['action'] == 'repay':
                    daily_net_value[date] = daily_net_value.get(date, 0) + amount_usd
                elif tx['action'] == 'redeemUnderlying':
                    daily_net_value[date] = daily_net_value.get(date, 0) - amount_usd
        
        sorted_dates = sorted(daily_net_value.keys())
        pnl_values = [daily_net_value[d] for d in sorted_dates]

        if len(pnl_values) > 1:
            returns = np.diff(pnl_values) / (np.abs(pnl_values[:-1]) + 1e-10)
            portfolio_data['returns'] = returns.tolist()
        else:
            portfolio_data['returns'] = [0]
        
        current_positions = {}
        for tx in txs:
            if 'actionData' in tx and 'assetSymbol' in tx['actionData']:
                asset = tx['actionData']['assetSymbol']
                amount_usd = self.get_usd_value_for_tx(tx)
                if tx['action'] in ['deposit', 'redeemUnderlying']:
                    current_positions[asset] = current_positions.get(asset, 0) + (amount_usd if tx['action'] == 'deposit' else -amount_usd)
                elif tx['action'] in ['borrow', 'repay']:
                    current_positions[asset] = current_positions.get(asset, 0) + (amount_usd if tx['action'] == 'repay' else -amount_usd)
        portfolio_data['positions'] = current_positions

        return portfolio_data

    @staticmethod
    def get_usd_value_for_tx(tx: Dict) -> float:
        try:
            if 'actionData' in tx:
                amount = float(tx['actionData'].get('amount', 0))
                price = float(tx['actionData'].get('assetPriceUSD', 0))
                asset_symbol = tx['actionData'].get('assetSymbol', '')
                if asset_symbol in ['USDC', 'USDT', 'DAI']:
                    return (amount / 1e6) * price 
                else:
                    return (amount / 1e18) * price
            return 0
        except (ValueError, TypeError):
            return 0

    def visualize_engineered_features(self, features_df: pd.DataFrame, num_plots: int = 5):
        if features_df.empty:
            print("No features to visualize.")
            return

        print(f"Generating plots for engineered features...")

        numerical_features = features_df.select_dtypes(include=np.number).columns.tolist()

        if not numerical_features:
            print("No numerical features to plot.")
            return

        features_variance = features_df[numerical_features].var().sort_values(ascending=False)
        features_to_plot = features_variance.index[:num_plots].tolist()

        if not features_to_plot:
            print("No suitable numerical features found for plotting.")
            return

        fig_hist, axes_hist = plt.subplots(len(features_to_plot), 1, figsize=(10, 4 * len(features_to_plot)))
        fig_hist.suptitle('Histograms of Key Engineered Features', y=1.02)

        if len(features_to_plot) == 1:
            axes_hist = [axes_hist]

        for i, feature in enumerate(features_to_plot):
            ax = axes_hist[i]
            plot_data = features_df[feature].dropna()
            if not plot_data.empty:
                ax.hist(plot_data, bins=30, edgecolor='black', alpha=0.7)
                ax.set_title(f'Distribution of {feature}')
                ax.set_xlabel(feature)
                ax.set_ylabel('Frequency')
            else:
                ax.set_title(f'No data to plot for {feature}')
            ax.grid(axis='y', alpha=0.75)

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig('engineered_features_histograms.png')
        print(f"Generated 'engineered_features_histograms.png' showing distributions of key features.")

        fig_scatter = None # Initialize fig_scatter outside try block

        if len(features_to_plot) > 1 and len(features_to_plot) <= 4:
            try:
                fig_scatter, ax_scatter = plt.subplots(figsize=(12, 12)) 
                pd.plotting.scatter_matrix(features_df[features_to_plot], diagonal='kde', ax=ax_scatter)
                plt.suptitle('Scatter Matrix of Key Engineered Features', y=0.98)
                plt.savefig('engineered_features_scatter_matrix.png')
                print("Generated 'engineered_features_scatter_matrix.png'.")
            except Exception as e:
                print(f"Could not generate scatter matrix: {e}")
            finally:
                if fig_scatter is not None: # Close figure only if it was successfully created
                    plt.close(fig_scatter)