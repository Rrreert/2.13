import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(layout="wide")

model1 = joblib.load('models/CPH.pkl')
model2 = joblib.load('models/Deepsurv.pkl')


@st.cache(show_spinner=False)
def load_setting():
    settings = {
        'Age': {'values': ["<35", "35-65", ">65"], 'type': 'selectbox', 'init_value': 0, 'add_after': ''},
        'Sex': {'values': ["Male", "Female"], 'type': 'radio', 'init_value': 0, 'add_after': ''},
        'Race': {'values': ["Black", "Other", "White"], 'type': 'selectbox', 'init_value': 0, 'add_after': ''},
        'Marital': {'values': ["Married", "Other"], 'type': 'radio', 'init_value': 0, 'add_after': ''},
        'Tumor_size': {'values': ["<11.8", ">=11.8", "Unknown"], 'type': 'selectbox', 'init_value': 0,
                       'add_after': ', cm'},
        'T_stage': {'values': ["T1", "T2", "T3", "TX"], 'type': 'selectbox', 'init_value': 0, 'add_after': ''},
        'N_stage': {'values': ["N0", "N1", "NX"], 'type': 'selectbox', 'init_value': 0, 'add_after': ''},
        'M_stage': {'values': ["M0", "M1", "M1a", "M1b"], 'type': 'selectbox', 'init_value': 0, 'add_after': ''},
        'Tumor_number': {'values': ["Single", "Multiple"], 'type': 'radio', 'init_value': 0, 'add_after': ''},
        'Grade': {'values': ["I-II", "III-IV", "Unknown"], 'type': 'selectbox', 'init_value': 0, 'add_after': ''},

        'SEER_stage': {'values': ["Distant", "Localized", "Regional", "Unknown/unstaged"], 'type': 'selectbox',
                       'init_value': 0, 'add_after': ''},
        'Radiation': {'values': ["No/Unknown", "Yes"], 'type': 'radio', 'init_value': 0, 'add_after': ''},
        'Chemotherapy': {'values': ["No/Unknown", "Yes"], 'type': 'radio', 'init_value': 0, 'add_after': ''},
        'Surgery': {'values': ["Radical", "No", "Other_Surgery"], 'type': 'selectbox',
                    'init_value': 0,
                    'add_after': ''},

        'Model': {'values': ["CPH", "Deepsurv"], 'type': 'selectbox',
                  'init_value': 0, 'add_after': ''},
    }
    input_keys = ['Age', 'Sex', 'Race', 'Marital', 'Tumor_size', 'Tumor_number',
                  "T_stage", "N_stage", "M_stage", "Grade", "SEER_stage", "Radiation", "Surgery",
                  "Chemotherapy", 'Model']
    return settings, input_keys


settings, input_keys = load_setting()


def load_model(val):
    model = ''
    if val == 'CPH':
        model = model1
    elif val == 'Deepsurv':
        model = model2
    return model


def get_code():
    sidebar_code = []
    for key in settings:
        if settings[key]['type'] == 'slider':
            sidebar_code.append(
                "{} = st.slider('{}',{},{},key='{}')".format(
                    key.replace(' ', '____'),
                    key + settings[key]['add_after'],
                    # settings[key]['values'][0],
                    ','.join(['{}'.format(value) for value in settings[key]['values']]),
                    settings[key]['init_value'],
                    key
                )
            )
        if settings[key]['type'] == 'selectbox':
            sidebar_code.append('{} = st.selectbox("{}",({}),{},key="{}")'.format(
                key.replace(' ', '____'),
                key + settings[key]['add_after'],
                ','.join('"{}"'.format(value) for value in settings[key]['values']),
                settings[key]['init_value'],
                key
            )
            )
        if settings[key]['type'] == 'radio':
            sidebar_code.append('{} = st.radio("{}",({}),{},key="{}")'.format(
                key.replace(' ', '____'),
                key + settings[key]['add_after'],
                ','.join('"{}"'.format(value) for value in settings[key]['values']),
                settings[key]['init_value'],
                key
            )
            )
    return sidebar_code


sidebar_code = get_code()

if 'patients' not in st.session_state:
    st.session_state['patients'] = []
if 'display' not in st.session_state:
    st.session_state['display'] = 1


def plot_survival():
    pd_data = pd.concat(
        [
            pd.DataFrame(
                {
                    'Survival': item['survival'],
                    'Time': item['times'],
                    'Patients': [item['No'] for i in item['times']]
                }
            ) for item in st.session_state['patients']
        ]
    )
    # 画图
    if st.session_state['display']:
        fig = px.line(pd_data, x="Time", y="Survival", color='Patients', range_y=[0, 1])
    else:
        fig = px.line(pd_data.loc[pd_data['Patients'] == pd_data['Patients'].to_list()[-1], :], x="Time", y="Survival",
                      range_y=[0, 1])
    fig.update_layout(title={
        'text': 'Estimated Survival Probability',
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(
            size=25
        )
    },
        # 背景颜色设置
        plot_bgcolor="LightGrey",
        xaxis_title="Time, month",
        yaxis_title="Survival probability",
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_patients():
    patients = pd.concat(
        [
            pd.DataFrame(
                dict(
                    {
                        'Patients': [item['No']],
                        '3-Year': ["{:.2f}%".format(item['3-year'] * 100)],
                        '5-Year': ["{:.2f}%".format(item['5-year'] * 100)],
                        '8-Year': ["{:.2f}%".format(item['8-year'] * 100)]
                    },
                    **item['arg']
                )
            ) for item in st.session_state['patients']
        ]
    ).reset_index(drop=True)
    st.dataframe(patients)


# @st.cache(show_spinner=True)
def predict():
    print('update patients . ##########')
    model = load_model(st.session_state["Model"])
    dic = {}
    for key in input_keys[:-1]:
        value = st.session_state[key]
        dic['{}_{}'.format(key, value)] = 1
    test_df = pd.DataFrame(columns=['Age_35-65', 'Age_<35', 'Age_>65', 'Chemotherapy_No/Unknown',
                                    'Chemotherapy_Yes', 'Grade_I-II', 'Grade_III-IV', 'Grade_Unknown',
                                    'M_stage_M0', 'M_stage_M1', 'M_stage_M1a', 'M_stage_M1b',
                                    'Marital_Married', 'Marital_Other', 'N_stage_N0', 'N_stage_N1',
                                    'N_stage_NX', 'Race_Black', 'Race_Other', 'Race_White',
                                    'Radiation_None/Unknown', 'Radiation_Yes', 'SEER_stage_Distant',
                                    'SEER_stage_Localized', 'SEER_stage_Regional',
                                    'SEER_stage_Unknown/unstaged', 'Sex_Female', 'Sex_Male', 'Surgery_No',
                                    'Surgery_Other_surgery', 'Surgery_Radical', 'T_stage_T1', 'T_stage_T2',
                                    'T_stage_T3', 'T_stage_TX', 'Tumor_number_Multiple',
                                    'Tumor_number_Single', 'Tumor_size_<11 8', 'Tumor_size_>=11 8',
                                    'Tumor_size_Unknown', 'status'])

    test_df = pd.concat([test_df, pd.DataFrame([dic])])

    test_df.fillna(0, inplace=True)

    survival = model.predict_survival(test_df)
    # x_axis = []
    # for inx in survival[0].x:
    #     if inx <= 155:
    #         x_axis.append(inx)
    #     else:
    #         break
    data = {
        'survival': survival,
        'times': [i for i in range(1, len(survival) + 1)],
        'No': len(st.session_state['patients']) + 1,
        'arg': {key: st.session_state[key] for key in input_keys},
        '3-year': survival[35],
        '5-year': survival[59],
        '8-year': survival[95],
    }
    st.session_state['patients'].append(
        data
    )
    print('update patients ... ##########')


def plot_below_header():
    col1, col2 = st.columns([1, 9])
    col3, col4, col5, col6, col7 = st.columns([2, 2, 2, 2, 2])
    with col1:
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        # st.session_state['display'] = ['Single', 'Multiple'].index(
        #     st.radio("Display", ('Single', 'Multiple'), st.session_state['display']))
        st.session_state['display'] = ['Single', 'Multiple'].index(
            st.radio("Display", ('Single', 'Multiple'), st.session_state['display']))
        # st.radio("Model", ('DeepSurv', 'NMTLR','RSF','CoxPH'), 0,key='model',on_change=predict())
    with col2:
        plot_survival()
    with col4:
        st.metric(
            label='3-Year survival probability',
            value="{:.2f}%".format(st.session_state['patients'][-1]['3-year'] * 100)
        )
    with col5:
        st.metric(
            label='5-Year survival probability',
            value="{:.2f}%".format(st.session_state['patients'][-1]['5-year'] * 100)
        )
    with col6:
        st.metric(
            label='8-Year survival probability',
            value="{:.2f}%".format(st.session_state['patients'][-1]['8-year'] * 100)
        )
    st.write('')
    st.write('')
    st.write('')
    plot_patients()
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')


st.header('Deep model for predicting survival of lung papillary adenocarcinoma',
          anchor='Cancer-specific survival of osteosarcoma')
if st.session_state['patients']:
    plot_below_header()

with st.sidebar:
    with st.form("my_form", clear_on_submit=False):
        for code in sidebar_code:
            exec(code)
        col8, col9, col10 = st.columns([3, 4, 3])
        with col9:
            prediction = st.form_submit_button(
                'Predict',
                on_click=predict,
                # args=[{key: eval(key.replace(' ', '____')) for key in input_keys}]
            )
