import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from PIL import Image
from data_preprocessing import CarDataPreprocessor
from prediction import predict_price

# Page config
st.set_page_config(
    page_title="🏎️ Premium Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

BENCHMARK_TOTAL_100K_KM = 1034594  # ₹1,034,594 for 100,000 km
BENCHMARK_PER_KM = 10.35  # ₹/km


@st.cache_resource
def load_model():
    """Load trained model and preprocessor"""
    try:
        pipeline = joblib.load('car_price_model.pkl')
        preprocessor_path = 'preprocessor.pkl'
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
            pipeline['preprocessor'] = preprocessor
        return pipeline
    except:
        st.warning("Model not found. Train first!")
        return None

@st.cache_data
def load_dataset():
    """Load dataset once"""
    df = pd.read_csv('cardekho_dataset.csv')
    return df

@st.cache_data
def get_unique_brands(_df=load_dataset()):
    """Get unique brands from cached data"""
    return sorted(_df['brand'].unique())[:20]

def main():
    # Custom CSS for premium look
    st.markdown("""
    <style>
        .main-header {
            font-size: 3.5rem !important;
            color: #1f77b4 !important;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        .stPlotlyChart {
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .block-container {
            padding-top: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Global Plotly config
    config = {'displayModeBar': True, 'displaylogo': False}

    st.title('Car Price Predictor')

    # Sidebar info
    with st.sidebar:
        st.header("📊 Quick Info")
        st.metric("Dataset Size", "15K+ cars")
        st.metric("Avg Price", "₹7.5L")
        st.metric("Top Brand", "Maruti")
        st.info("Built with XGBoost | R²: 0.90+")

    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict Price", "⚖️ Compare Cars (AI Rec)", "📈 Insights", "🤖 AI Car Suggester"])

    with tab1:
        st.header("🔮 Single Car Price Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 Car Specifications")
            preprocessor = CarDataPreprocessor()
            preprocessor.create_preprocessing_pipeline()
            feature_names = preprocessor.numerical_features + preprocessor.categorical_features
            
            # Numerical sliders (realistic ranges from data)
            vehicle_age = st.slider("Vehicle Age (years)", 0, 20, 5)
            km_driven = st.slider("KM Driven", 0, 500000, 50000)
            mileage = st.slider("Mileage (kmpl)", 8.0, 35.0, 18.0)
            engine = st.slider("Engine (cc)", 600, 4000, 1200)
            max_power = st.slider("Max Power (bhp)", 40, 300, 80)
            seats = st.select_slider("Seats", [4, 5, 6, 7, 8, 9], 5)
            
        with col2:
            st.subheader("🔧 Categorical")
            seller_type = st.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
            fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG"])
            transmission_type = st.selectbox("Transmission", ["Manual", "Automatic"])
            brand = st.selectbox("Brand", get_unique_brands())
        
        if st.button("🚀 Predict Price", type="primary"):
            with st.spinner("Predicting..."):
                price = predict_price(
                    vehicle_age=vehicle_age, km_driven=km_driven, seller_type=seller_type,
                    fuel_type=fuel_type, transmission_type=transmission_type,
                    mileage=mileage, engine=engine, max_power=max_power, seats=seats, brand=brand
                )
                
                # Cost per km calculations
                predicted_cost_100k = price / km_driven * 100000 if km_driven > 0 else 0
                predicted_per_km = predicted_cost_100k / 100000
                savings_pct = max(0, (BENCHMARK_PER_KM - predicted_per_km) / BENCHMARK_PER_KM * 100)
                
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("Predicted Price", f"₹{price:,.0f}", delta="±10%")
            with col_b:
                st.metric("Pred Cost/100k km", f"₹{predicted_cost_100k:,.0f}")
            with col_c:
                st.metric("Benchmark", f"₹{BENCHMARK_TOTAL_100K_KM:,}/100k km")
            with col_d:
                st.metric("Savings", f"+{savings_pct:.1f}%", delta=f"vs ₹10.35/km")
            
            st.info(f"**Benchmark**: ₹{BENCHMARK_TOTAL_100K_KM:,} total cost for 100k km (₹10.35/km). Your predicted cost: ₹{predicted_per_km:.2f}/km.")
            st.success("🎉 Prediction complete!")
            
            # Simple feature impact viz
            st.subheader("📊 Feature Impact")
            impact_data = {
                'Feature': ['Vehicle Age', 'KM Driven', 'Max Power', 'Mileage'],
                'Impact': [-vehicle_age*0.12, -km_driven/1e5, max_power*0.015, mileage*0.08]
            }
            fig = px.bar(
                pd.DataFrame(impact_data), 
                x='Impact', y='Feature', orientation='h', 
                color='Impact',
                color_continuous_scale='RdYlBu_r',
                title="🚀 Feature Impact on Price (Higher = More Influence)"
            ).update_layout(template='plotly_white', font_size=12, height=400)
            fig.add_annotation(x=-0.1, y=0, text="Negative = Price ↓", showarrow=False, font_size=11)
            st.plotly_chart(fig, use_container_width=True, config=config)

    with tab2:
        st.header("⚖️ Compare Two Cars - AI Recommendation")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("🚗 First Car")
            age_a = st.slider("Age ", 0 , 20, 5, key="a1")
            km_a = st.slider("KM Driven", 0, 500000, 50000, key="a2")
            brand_a = st.selectbox("Brand", get_unique_brands(), key="a3")
            seller_a = st.selectbox("Seller", ["Individual", "Dealer"], key="a4")
            fuel_a = st.selectbox("Fuel", ["Petrol", "Diesel"], key="a5")
            trans_a = st.selectbox("Trans", ["Manual", "Automatic"], key="a6")
            mileage_a = st.slider("Mileage", 8.0, 35.0, 18.0, key="a7")
            engine_a = st.slider("Engine", 600, 4000, 1200, key="a8")
            power_a = st.slider("Power", 40, 300, 80, key="a9")
            seats_a = st.select_slider("Seats", [4,5,6,7,8], 5, key="a10")
        
        with col_right:
            st.subheader("🚙 Second Car")
            age_b = st.slider("", 0, 20, 3, key="b1")
            km_b = st.slider("", 0, 500000, 30000, key="b2")
            brand_b = st.selectbox("", get_unique_brands(), key="b3")
            seller_b = st.selectbox("", ["Individual", "Dealer"], key="b4")
            fuel_b = st.selectbox("", ["Petrol", "Diesel"], key="b5")
            trans_b = st.selectbox("", ["Manual", "Automatic"], key="b6")
            mileage_b = st.slider("", 8.0, 35.0, 20.0, key="b7")
            engine_b = st.slider("", 600, 4000, 1400, key="b8")
            power_b = st.slider("", 40, 300, 90, key="b9")
            seats_b = st.select_slider("", [4,5,6,7,8], 5, key="b10")
        
        if st.button("🤖 Compare & Recommend", type="primary"):
            with st.spinner("Analyzing..."):
                price_a = predict_price(vehicle_age=age_a, km_driven=km_a, seller_type=seller_a, 
                                       fuel_type=fuel_a, transmission_type=trans_a, mileage=mileage_a,
                                       engine=engine_a, max_power=power_a, seats=seats_a, brand=brand_a)
                price_b = predict_price(vehicle_age=age_b, km_driven=km_b, seller_type=seller_b, 
                                       fuel_type=fuel_b, transmission_type=trans_b, mileage=mileage_b,
                                       engine=engine_b, max_power=power_b, seats=seats_b, brand=brand_b)

            
            df_compare = pd.DataFrame({
                'Metric': ['Price', 'Age (yr)', 'KM Driven', 'Value Score'],
                'Car A': [f"₹{price_a:,.0f}", age_a, km_a, f"{100-age_a*5:.0f}%"],
                'Car B': [f"₹{price_b:,.0f}", age_b, km_b, f"{100-age_b*5:.0f}%"]
            })
            st.dataframe(df_compare)
            
            # AI Rec
            value_a = price_a / (1 + age_a + km_a/1e5)
            value_b = price_b / (1 + age_b + km_b/1e5)
            rec = "Car A" if value_a < value_b else "Car B"
            st.success(f"**🎯 Recommendation: Buy {rec}!** It's {abs(value_a-value_b)/min(value_a,value_b)*100:.1f}% better value.")
            
            # Radar chart
            fig = go.Figure(data=[
                go.Scatterpolar(r=[age_a, km_a/1e5, price_a/1e6, 95-age_a*4], 
                               theta=['Age (yr)', 'KM/100k', 'Price (₹M)', 'Condition Score'], 
                               fill='toself', name='Car A', line_color='darkblue'),
                go.Scatterpolar(r=[age_b, km_b/1e5, price_b/1e6, 95-age_b*4], 
                               theta=['Age (yr)', 'KM/100k', 'Price (₹M)', 'Condition Score'], 
                               fill='toself', name='Car B', line_color='darkgreen')
            ])
            fig.update_layout(
                template='plotly_white',
                polar=dict(radialaxis=dict(visible=True, range=[0, max(price_a/1e6*1.2, price_b/1e6*1.2)])),
                title="⚖️ Car Comparison Radar - Lower Age/KM/Price = Better Deal",
                height=450, font_size=12
            )
            st.plotly_chart(fig, use_container_width=True, config=config)

    with tab3:
        st.header("📈 Insights & Visualizations")
        df = load_dataset()
        
        st.subheader("🔍 Key Insights Dashboard", divider=True)
        
        # 2x2 Subplots for neat layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('📊 Price Distribution (Log Scale)', '📉 Age vs Price by Brand', 
                           '🔥 Top 10 Brands Avg Price (₹L)', '📈 Correlation Heatmap'),
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "heatmap"}]]
        )
        
        # Price histogram with log scale
        fig.add_trace(
            px.histogram(df, x='selling_price', nbins=50, 
                        color_discrete_sequence=['#1f77b4']).data[0],
            row=1, col=1
        )
        fig.update_xaxes(type='log', row=1, col=1)
        
        # Age vs price scatter
        scatter = px.scatter(df, x='vehicle_age', y='selling_price', color='brand', 
                            color_discrete_sequence=px.colors.qualitative.Set3).data[0]
        fig.add_trace(scatter, row=1, col=2)
        fig.update_yaxes(type='log', row=1, col=2)
        
        # Top brands bar
        brand_avg = df.groupby('brand')['selling_price'].mean().sort_values(ascending=False).head(10) / 1e5
        bar = px.bar(x=brand_avg.index, y=brand_avg.values, 
                    color=brand_avg.values, color_continuous_scale='Viridis').data[0]
        fig.add_trace(bar, row=2, col=1)
        fig.update_yaxes(title='Avg Price (₹ Lakh)', row=2, col=1)
        
        # Correlation heatmap
        numeric_df = df.select_dtypes(include=[np.number]).corr()
        heatmap = px.imshow(numeric_df, color_continuous_scale='RdBu_r', aspect='auto').data[0]
        fig.add_trace(heatmap, row=2, col=2)
        
        fig.update_layout(
            template='plotly_white', height=800, showlegend=False, font_size=11,
            title_text="🏎️ CarDekho Dataset Insights - Neat & Interactive", 
            title_font_size=18, title_x=0.5
        )
        fig.add_annotation(text="💡 Hover for details | Log scales for better distribution view", 
                          xref="paper", yref="paper", x=0, y=-0.05, showarrow=False)
        
        st.plotly_chart(fig, use_container_width=True, config=config)
        
        with st.expander("📈 Additional Stats"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric('Total Cars', f"{len(df):,}")
            with col2:
                st.metric('Avg Price', f"₹{df['selling_price'].mean():,.0f}")
            with col3:
                st.metric('Median Age', f"{df['vehicle_age'].median():.1f} yrs")
            with col4:
                st.metric('Top Fuel', df['fuel_type'].mode().iloc[0])

    with tab4:
        st.header("🤖 AI Car Suggester")
        
        df = load_dataset()
        
        st.info(f"📊 {len(df):,} cars available for suggestions")
        
        # User needs inputs
        col1, col2 = st.columns(2)
        with col1:
            max_budget = st.slider("💰 Max Budget (₹)", 100000, 10000000, 500000)
            pref_fuel = st.multiselect("⛽ Preferred Fuel", df['fuel_type'].unique(), default=['Petrol', 'Diesel'])

            min_mileage = st.slider("⚡ Min Mileage (kmpl)", 10.0, 30.0, 15.0)
        with col2:
            min_seats = st.slider("👨‍👩‍👧‍👦 Min Seats", 4, 9, 5)
            pref_trans = st.multiselect("⚙️ Transmission", df['transmission_type'].unique(), default=['Manual'])
            min_power = st.slider("💪 Min Power (bhp)", 50, 250, 70)
            pref_brands = st.multiselect("🏷️ Preferred Brands", df['brand'].unique()[:15], default=[])
        
        if st.button("🔍 Find Best Matches", type="primary"):
            with st.spinner("🤖 AI analyzing needs & predicting prices..."):
                # Filter dataset
                filtered = df[
                    (df['selling_price'] <= max_budget * 1.1) &
                    (df['mileage'] >= min_mileage) &
                    (df['seats'] >= min_seats) &
                    (df['max_power'] >= min_power) &
                    (df['fuel_type'].isin(pref_fuel)) &
                    (df['transmission_type'].isin(pref_trans))
                ]
                if pref_brands:
                    filtered = filtered[filtered['brand'].isin(pref_brands)]
                
                if len(filtered) == 0:
                    st.warning("😅 No cars match your criteria. Try relaxing filters!")
                else:
                    # Predict prices for filtered cars
                    predictions = []
                    for idx, row in filtered.iterrows():
                        try:
                            pred_price = predict_price(**row.to_dict())
                            score = (
                                100 / (1 + row['vehicle_age'] + row['km_driven']/1e5) *  # Condition
                                (row['mileage'] / 20) *  # Efficiency
                                np.exp(- (pred_price - max_budget/2)**2 / (max_budget**2 / 4))  # Budget fit
                            )
                            predictions.append({
                                'index': idx, 'pred_price': pred_price, 'value_score': score,
                                **row.to_dict()
                            })
                        except:
                            continue
                    
                    if predictions:
                        top3 = sorted(predictions, key=lambda x: x['value_score'], reverse=True)[:3]
                        
                        for i, car in enumerate(top3, 1):
                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                st.markdown(f"**#{i} {car['brand']} {car['model']}**")
                                st.caption(f"Age: {car['vehicle_age']}yr | KM: {car['km_driven']/1000:.0f}k | Mileage: {car['mileage']:.1f}kmpl")
                            with col2:
                                st.metric("Pred Price", f"₹{car['pred_price']:,.0f}")
                            with col3:
                                st.metric("Value Score", f"{car['value_score']:.0f}%")
                            
                            # AI Explanation
                            explains = []
                            if car['vehicle_age'] <= 3: explains.append("🆕 Young age - low risk & good condition")
                            if car['mileage'] >= 20: explains.append("⚡ Excellent mileage - fuel savings")
                            if car['pred_price'] <= max_budget * 0.8: explains.append("💰 Great deal - under budget")
                            if car['seats'] >= 5: explains.append("👨‍👩‍👧‍👦 Family-friendly seating")
                            st.success("🤖 **Why this car?** " + " | ".join(explains[:3]))
                            
                            # Picture
                            img_query = f"{car['brand']}-{car.get('model', 'car').replace(' ', '-')}"
                            st.image(f"https://source.unsplash.com/400x250/?{img_query}", width="stretch")
                        
                        # Viz top3
                        top_df = pd.DataFrame([{
                            'Car': f"#{i+1} {c['brand']} {c['model'][:20]}",
                            'Pred Price (₹L)': c['pred_price']/1e5,
                            'Value Score (%)': c['value_score']
                        } for i, c in enumerate(top3)])
                        fig = px.bar(top_df, x='Car', y=['Pred Price (₹L)', 'Value Score (%)'], 
                                   barmode='group', 
                                   title="🏆 Top 3 Recommendations - Balance Price vs Value",
                                   color_discrete_map={'Pred Price (₹L)': '#ef5532', 'Value Score (%)': '#00cc96'})
                        fig.update_layout(template='plotly_white', height=400, font_size=11)
                        st.plotly_chart(fig, use_container_width=True, config=config)
                    else:
                        st.info("No predictions possible. Model may need training.")

    # Footer
    st.markdown("---")
    st.markdown("")

if __name__ == "__main__":
    main()

