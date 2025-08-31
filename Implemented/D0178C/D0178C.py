# DO-178C AI Certification Suite Page
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def do178c_ai_certification_page():
    st.set_page_config(page_title="DO-178C AI Certification Suite", layout="wide")

    st.sidebar.title("📂 DO-178C AI Certification Workflows")
    workflow = st.sidebar.selectbox(
        "Select Workflow",
        [
            "Requirements Traceability",
            "Data Validation",
            "Verification & Validation",
            "Compliance Reporting",
            "Emergent Behavior Simulation",
            "Emergent Behavior Control",
            "Sensor Fusion Validation"
        ]
    )

    # -------------------------
    # 1. Requirements Traceability
    # -------------------------
    def requirements_traceability():
        st.title("📝 Requirements Traceability Workflow")
        st.markdown("""
        Link high-level system requirements to AI model specifications and maintain traceability to design, code, and tests.
        """)
        uploaded_file = st.file_uploader("Upload Requirements CSV", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())
            if "Design_Link" in df.columns and "Test_Link" in df.columns:
                df['Traceable'] = df['Design_Link'].notna() & df['Test_Link'].notna()
                st.subheader("Traceability Check")
                st.dataframe(df[['Requirement_ID','Traceable']])
            else:
                st.warning("CSV must include 'Design_Link' and 'Test_Link' columns")

    # -------------------------
    # 2. Data Validation
    # -------------------------
    def data_validation():
        st.title("🗂️ Data Validation Workflow")
        st.markdown("""
        Ensure datasets meet quality standards for AI certification:
        - Missing value analysis
        - Outlier detection
        - Feature distribution checks
        """)
        uploaded_file = st.file_uploader("Upload Dataset CSV", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write(df.head())
            st.subheader("Missing Values")
            st.dataframe(df.isna().sum())
            st.subheader("Outlier Detection (IQR)")
            numeric_cols = df.select_dtypes(include=np.number).columns
            outliers = {}
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers[col] = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)][col].count()
            st.dataframe(pd.DataFrame.from_dict(outliers, orient='index', columns=['Outliers']))

    # -------------------------
    # 3. Verification & Validation
    # -------------------------
    def verification_validation():
        st.title("✅ Verification & Validation Workflow")
        uploaded_file = st.file_uploader("Upload Test Results CSV", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())
            if "Test_Pass" in df.columns:
                pass_count = df['Test_Pass'].sum()
                fail_count = len(df) - pass_count
                st.write(f"✅ Pass: {pass_count}, ❌ Fail: {fail_count}")
                st.bar_chart(df['Test_Pass'].value_counts())
            else:
                st.warning("CSV must include a 'Test_Pass' boolean column")

    # -------------------------
    # 4. Compliance Reporting
    # -------------------------
    def compliance_reporting():
        st.title("📊 Compliance Reporting")
        uploaded_req = st.file_uploader("Upload Requirements Traceability CSV", type="csv", key="req")
        uploaded_test = st.file_uploader("Upload Test Results CSV", type="csv", key="test")
        if uploaded_req and uploaded_test:
            req_df = pd.read_csv(uploaded_req)
            test_df = pd.read_csv(uploaded_test)
            # Traceability coverage
            traceable_count = req_df['Traceable'].sum() if 'Traceable' in req_df.columns else 0
            total_req = len(req_df)
            st.write(f"Traceability Coverage: {traceable_count}/{total_req} ({traceable_count/total_req*100:.2f}%)")
            # Verification summary
            if 'Test_Pass' in test_df.columns:
                pass_count = test_df['Test_Pass'].sum()
                total_tests = len(test_df)
                st.write(f"Verification Pass Rate: {pass_count}/{total_tests} ({pass_count/total_tests*100:.2f}%)")
            st.success("Compliance report generated successfully!")

    # -------------------------
    # 5. Emergent Behavior Simulation
    # -------------------------
    def emergent_behavior_sim():
        st.title("🦅 Emergent Behavior Simulation (Flocking / Boids)")
        st.markdown("Demonstrates emergent swarm behavior from local agent rules.")
        n_agents = st.slider("Number of Agents", 10, 200, 50)
        n_steps = st.slider("Number of Steps", 10, 100, 30)
        np.random.seed(42)
        positions = np.random.rand(n_agents, 2) * 100
        velocities = (np.random.rand(n_agents, 2) - 0.5) * 2.0
        def update(positions, velocities):
            for i in range(n_agents):
                center = np.mean(positions, axis=0)
                cohesion = (center - positions[i]) * 0.05
                alignment = (np.mean(velocities, axis=0) - velocities[i]) * 0.05
                diff = positions[i] - positions
                dist = np.linalg.norm(diff, axis=1)
                separation = np.sum(diff[dist<5], axis=0) * 0.05 if np.any(dist<5) else 0
                velocities[i] += cohesion + alignment + separation
                speed = np.linalg.norm(velocities[i])
                if speed > 2.0: velocities[i] = velocities[i]/speed*2.0
            positions += velocities
            return positions, velocities
        fig, ax = plt.subplots()
        for _ in range(n_steps):
            positions, velocities = update(positions, velocities)
            ax.clear()
            ax.scatter(positions[:,0], positions[:,1], c='blue')
            ax.set_xlim(0,100); ax.set_ylim(0,100)
            ax.set_title("Emergent Flocking Simulation")
        st.pyplot(fig)

    # -------------------------
    # 6. Emergent Behavior Control
    # -------------------------
    def emergent_behavior_control():
        st.title("🧭 Emergent Behavior Control (Leader-Follower / Attractor)")
        st.markdown("Demonstrates control strategies for emergent swarms via leaders or attractors.")
        n_agents = st.slider("Number of Agents", 10, 200, 50)
        n_leaders = st.slider("Number of Leaders", 1, 5, 1)
        n_steps = st.slider("Number of Steps", 10, 100, 30)
        np.random.seed(42)
        positions = np.random.rand(n_agents,2)*100
        leaders = np.random.choice(range(n_agents), n_leaders, replace=False)
        target = np.array([80,80])
        def update(positions):
            new_positions = positions.copy()
            for i in range(n_agents):
                if i in leaders:
                    new_positions[i] += (target - positions[i]) * 0.1
                else:
                    neighbors = np.linalg.norm(positions-positions[i],axis=1)<10
                    if np.sum(neighbors)>1:
                        center = np.mean(positions[neighbors],axis=0)
                        new_positions[i] += (center - positions[i]) * 0.05
            return new_positions
        fig, ax = plt.subplots()
        for _ in range(n_steps):
            positions = update(positions)
            ax.clear()
            ax.scatter(positions[:,0], positions[:,1], c='blue')
            ax.scatter(positions[leaders,0], positions[leaders,1], c='red')
            ax.scatter(target[0],target[1], c='green', marker='*', s=200)
            ax.set_xlim(0,100); ax.set_ylim(0,100)
            ax.set_title("Emergent Behavior Control Simulation")
        st.pyplot(fig)

    # -------------------------
    # 7. Sensor Fusion Validation
    # -------------------------
    def sensor_fusion_validation():
        st.title("🛰️ Sensor Fusion Validation Workflow")
        st.markdown("Demonstrates Kalman Filter sensor fusion validation for AI/avionics systems.")
        n_steps = st.slider("Number of Time Steps", 10, 200, 50)
        measurement_var = st.slider("Measurement Variance", 0.01, 5.0, 1.0)
        np.random.seed(42)
        true_signal = np.cumsum(np.random.randn(n_steps))
        measurements = true_signal + np.random.normal(0,np.sqrt(measurement_var), n_steps)
        x_est = 0.0; P=1.0; Q=0.1; R=measurement_var; estimates=[]
        for z in measurements:
            P = P+Q
            K = P/(P+R)
            x_est = x_est + K*(z-x_est)
            P = (1-K)*P
            estimates.append(x_est)
        fig, ax = plt.subplots()
        ax.plot(true_signal,label="True Signal")
        ax.plot(measurements,'o',label="Measurements",alpha=0.5)
        ax.plot(estimates,label="Kalman Estimate")
        ax.legend(); ax.set_title("Kalman Filter Sensor Fusion")
        st.pyplot(fig)

    # -------------------------
    # Workflow Dispatcher
    # -------------------------
    workflow_dispatch = {
        "Requirements Traceability": requirements_traceability,
        "Data Validation": data_validation,
        "Verification & Validation": verification_validation,
        "Compliance Reporting": compliance_reporting,
        "Emergent Behavior Simulation": emergent_behavior_sim,
        "Emergent Behavior Control": emergent_behavior_control,
        "Sensor Fusion Validation": sensor_fusion_validation
    }

    # Execute selected workflow
    workflow_dispatch[workflow]()
