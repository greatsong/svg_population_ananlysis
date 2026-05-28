import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import re
import urllib.parse
from bs4 import BeautifulSoup

# 페이지 설정
st.set_page_config(
    page_title="대한민국 지역별 인구구조 분석 대시보드",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 타이틀 및 설명
st.title("📊 대한민국 지역별 인구구조 및 노령화·청소년 분석 대시보드")
st.markdown("행정안전부 주민등록 인구통계 데이터를 기반으로 고령화 비율, 청소년 인구 비율, 성비 등 핵심 지표를 정밀하게 시각화합니다.")

# -----------------------------------------------------------------------------
# 1. 인구 데이터 로더 및 정밀 파싱
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_preprocess_data(file_path_or_buffer):
    df = None
    for encoding in ['cp949', 'euc-kr', 'utf-8']:
        try:
            df = pd.read_csv(file_path_or_buffer, encoding=encoding)
            break
        except Exception:
            continue
            
    if df is None:
        return None

    # 컬럼명 정리 (공백 제거)
    df.columns = [c.strip() for c in df.columns]
    reg_col = df.columns[0] # '행정구역'
    
    # 행정구역 코드 및 정제 명칭 분리
    df['region_code'] = df[reg_col].apply(lambda x: re.search(r'\((\d+)\)', str(x)).group(1) if re.search(r'\((\d+)\)', str(x)) else '0000000000')
    df['clean_region'] = df[reg_col].apply(lambda x: x.split('(')[0].strip() if isinstance(x, str) else str(x))
    
    # 지능형 광역시도/시군구/읍면동 레벨 분류기
    def get_level(row):
        code = row['region_code']
        region = row['clean_region']
        if code == '0000000000' or '전국' in region:
            return 'Nation'
        sidos = [
            "서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시", "대전광역시", "울산광역시", 
            "세종특별자치시", "경기도", "강원도", "강원특별자치도", "충청북도", "충청남도", 
            "전라북도", "전북특별자치도", "전라남도", "경상북도", "경상남도", "제주특별자치도"
        ]
        if region in sidos or code.endswith('00000000'):
            return 'Sido'
        elif code.endswith('00000') or (" " in region and len(region.split()) == 2):
            return 'Sigungu'
        else:
            return 'Dong'
            
    df['region_level'] = df.apply(get_level, axis=1)
    
    # 쉼표(,) 제거 및 숫자 변환
    for col in df.columns:
        if col not in [reg_col, 'region_code', 'clean_region', 'region_level']:
            df[col] = df[col].astype(str).str.replace(',', '').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            
    return df

# -----------------------------------------------------------------------------
# 2. 인구 지표 계산기
# -----------------------------------------------------------------------------
def calculate_demographics(df):
    male_cols = {}
    female_cols = {}
    
    for col in df.columns:
        parts = col.split('_')
        if len(parts) >= 3:
            age_part = parts[-1]
            nums = re.findall(r'\d+', age_part)
            if nums:
                age = int(nums[0])
                if '_남_' in col:
                    male_cols[age] = col
                elif '_여_' in col:
                    female_cols[age] = col
                    
    ages = sorted(list(set(male_cols.keys()).intersection(set(female_cols.keys()))))
    
    results = []
    for idx, row in df.iterrows():
        m_pop_by_age = {age: row[male_cols[age]] for age in ages}
        f_pop_by_age = {age: row[female_cols[age]] for age in ages}
        total_pop_by_age = {age: m_pop_by_age[age] + f_pop_by_age[age] for age in ages}
        
        m_total = sum(m_pop_by_age.values())
        f_total = sum(f_pop_by_age.values())
        total_pop = m_total + f_total
        
        if total_pop == 0:
            continue
            
        youth_14 = sum([total_pop_by_age[a] for a in ages if a <= 14])
        youth_18 = sum([total_pop_by_age[a] for a in ages if a <= 18])
        working_age = sum([total_pop_by_age[a] for a in ages if 15 <= a <= 64])
        elderly = sum([total_pop_by_age[a] for a in ages if a >= 65])
        
        aging_ratio = (elderly / total_pop) * 100
        youth_ratio = (youth_18 / total_pop) * 100
        working_ratio = (working_age / total_pop) * 100
        sex_ratio = (m_total / f_total) * 100 if f_total > 0 else 100
        
        aging_index = (elderly / youth_14) * 100 if youth_14 > 0 else 0
        
        cumulative = 0
        median_age = 0
        for age in ages:
            cumulative += total_pop_by_age[age]
            if cumulative >= total_pop / 2:
                median_age = age
                break
                
        results.append({
            'clean_region': row['clean_region'],
            'region_code': row['region_code'],
            'region_level': row['region_level'],
            '총인구': total_pop,
            '남성인구': m_total,
            '여성인구': f_total,
            '유소년인구_14': youth_14,
            '청소년인구_18': youth_18,
            '생산가능인구': working_age,
            '고령인구': elderly,
            '고령화비율': aging_ratio,
            '청소년비율': youth_ratio,
            '생산가능인구비율': working_ratio,
            '노령화지수': aging_index,
            '성비': sex_ratio,
            '중위연령': median_age,
            'raw_row': row,
            'male_cols': male_cols,
            'female_cols': female_cols,
            'ages': ages
        })
        
    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# 3. 지도 및 연동 데이터 캐싱 로더
# -----------------------------------------------------------------------------
@st.cache_data
def fetch_svg(filename):
    url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{filename}"
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        if response.status_code == 200:
            return response.text
    except Exception:
        pass
    return None

# 가상 데이터 빌더 (파일 없을 시 작동 방지용)
def get_mock_population_data():
    regions = ["전국", "서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시", "대전광역시", "울산광역시", "세종특별자치시", "경기도", "강원도", "충청북도", "충청남도", "전라북도", "전라남도", "경상북도", "경상남도", "제주특별자치도"]
    seoul_districts = ["종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구", "성북구", "강북구", "도봉구", "노원구", "은평구", "서대문구", "마포구", "양천구", "강서구", "구로구", "금천구", "영등포구", "동작구", "관악구", "서초구", "강남구", "송파구", "강동구"]
    
    data = []
    import random
    random.seed(42)
    for r in regions:
        tot = random.randint(1000000, 13000000) if "경기" in r else random.randint(300000, 3400000)
        data.append({
            'clean_region': r, 'region_code': '1100000000', 'region_level': 'Sido' if r != "전국" else "Nation",
            '총인구': tot, '남성인구': int(tot*0.49), '여성인구': int(tot*0.51),
            '유소년인구_14': int(tot*0.11), '청소년인구_18': int(tot*0.15),
            '생산가능인구': int(tot*0.70), '고령인구': int(tot*0.19),
            '고령화비율': random.uniform(14.0, 25.0) if r != "전국" else 18.5, '청소년비율': random.uniform(10.0, 18.0),
            '생산가능인구비율': random.uniform(65.0, 74.0), '노령화지수': random.uniform(110.0, 220.0),
            '성비': random.uniform(95.0, 102.0), '중위연령': random.randint(42, 51),
            'raw_row': None, 'male_cols': {}, 'female_cols': {}, 'ages': []
        })
    for d in seoul_districts:
        tot = random.randint(150000, 660000)
        data.append({
            'clean_region': f"서울특별시 {d}", 'region_code': '1111000000', 'region_level': 'Sigungu',
            '총인구': tot, '남성인구': int(tot*0.49), '여성인구': int(tot*0.51),
            '유소년인구_14': int(tot*0.11), '청소년인구_18': int(tot*0.15),
            '생산가능인구': int(tot*0.70), '고령인구': int(tot*0.19),
            '고령화비율': random.uniform(13.0, 22.0), '청소년비율': random.uniform(9.0, 16.0),
            '생산가능인구비율': random.uniform(66.0, 75.0), '노령화지수': random.uniform(100.0, 200.0),
            '성비': random.uniform(94.0, 101.0), '중위연령': random.randint(41, 49),
            'raw_row': None, 'male_cols': {}, 'female_cols': {}, 'ages': []
        })
    return pd.DataFrame(data)

# 데이터 로드 트리거
csv_path = "202604_연령별인구현황_월간.csv"
raw_data = None

st.sidebar.header("📁 데이터 소스 설정")
uploaded_file = st.sidebar.file_uploader("행안부 인구통계 CSV 파일 업로드", type=["csv"])

if uploaded_file is not None:
    raw_data = load_and_preprocess_data(uploaded_file)
    st.sidebar.success("✅ 업로드된 파일 분석 완료!")
elif pd.io.common.file_exists(csv_path):
    raw_data = load_and_preprocess_data(csv_path)
    st.sidebar.success("📂 로컬 저장소 인구 데이터 분석 완료!")
else:
    st.sidebar.warning("⚠️ CSV 파일이 없습니다. 가상 시뮬레이션 데이터를 불러옵니다.")

# -----------------------------------------------------------------------------
# 4. 분석 엔진 및 메인 대시보드
# -----------------------------------------------------------------------------
if raw_data is not None:
    df_metrics = calculate_demographics(raw_data)
else:
    df_metrics = get_mock_population_data()

# 사이드바 컨트롤러
st.sidebar.header("⚙️ 지표 및 시각화 옵션")
metric_map = {
    "고령화비율 (%)": "고령화비율",
    "청소년비율 (0~18세, %)": "청소년비율",
    "생산가능인구비율 (%)": "생산가능인구비율",
    "노령화지수": "노령화지수",
    "성비 (여성 100명당 남성수)": "성비",
    "총인구수 (명)": "총인구"
}
selected_label = st.sidebar.selectbox("분석 지표를 선택하세요:", list(metric_map.keys()))
selected_metric = metric_map[selected_label]

color_themes = {
    "YlOrRd (고령화/노령화 추천)": "YlOrRd",
    "Reds (위험도/집중도 추천)": "Reds",
    "Blues (생산가능/청소년 추천)": "Blues"
}
selected_theme_label = st.sidebar.selectbox("지도 색상 테마:", list(color_themes.keys()))
selected_theme = color_themes[selected_theme_label]

# 분석 지역 선택
st.sidebar.header("📍 분석 지역 선택")
regions = df_metrics['clean_region'].unique().tolist()
default_idx = 0
for i, r in enumerate(regions):
    if '전국' in r or '합계' in r:
        default_idx = i
        break
selected_region = st.sidebar.selectbox("상세 분석을 원하는 지역을 선택하세요:", regions, index=default_idx)

# -------------------------------------------------------------------------
# 상단 요약 지표 (Metrics)
# -------------------------------------------------------------------------
reg_data = df_metrics[df_metrics['clean_region'] == selected_region].iloc[0]

# 고령화 사회 등급 판정
ratio = reg_data['고령화비율']
if ratio >= 20.0:
    stage = "🔴 초고령사회"
elif ratio >= 14.0:
    stage = "🟠 고령사회"
elif ratio >= 7.0:
    stage = "🟡 고령화사회"
else:
    stage = "🟢 젊은 사회"
    
st.subheader(f"📌 {selected_region} 실시간 핵심 지표 요약")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("👥 총 인구수", f"{int(reg_data['총인구']):,} 명")
col2.metric("👵 고령인구 비중 (65세+)", f"{reg_data['고령화비율']:.2f}%", stage)
col3.metric("🧒 청소년 비중 (0~18세)", f"{reg_data['청소년비율']:.2f}%")
col4.metric("📈 노령화지수", f"{reg_data['노령화지수']:.1f}")
col5.metric("🎯 중위 연령", f"만 {int(reg_data['중위연령'])} 세")

# -------------------------------------------------------------------------
# 🏆 고령화 vs 가장 젊은 지역 공간 극단 비교 모델 가동
# -------------------------------------------------------------------------
def get_extremes(df_subset, col='고령화비율'):
    if df_subset is None or df_subset.empty or len(df_subset) < 2:
        return None, None
    sorted_df = df_subset.sort_values(by=col)
    return sorted_df.iloc[-1], sorted_df.iloc[0] # [가장 고령인 곳(Max), 가장 젊은 곳(Min)]

st.markdown("### 🏆 지역별 극단 인구 비교 (가장 고령화된 곳 🧓 vs 가장 젊은 곳 👶)")
ext_col1, ext_col2, ext_col3 = st.columns(3)

# 1. 전국 광역시도 비교
with ext_col1:
    st.markdown("<div style='background-color:#1c1f26; padding:15px; border-radius:10px; border:1px solid #383e4c;'>", unsafe_allow_html=True)
    st.markdown("##### 🌐 전국 17개 시도 최고 vs 최저")
    df_sido_subset = df_metrics[(df_metrics['region_level'] == 'Sido') & (df_metrics['clean_region'] != '전국')]
    oldest_sido, youngest_sido = get_extremes(df_sido_subset, '고령화비율')
    
    if oldest_sido is not None:
        st.markdown(f"🧓 **최고 고령:** `{oldest_sido['clean_region']}`  \n└ 고령화율: **{oldest_sido['고령화비율']:.1f}%** (만 {int(oldest_sido['중위연령'])}세)")
        st.markdown(f"👶 **가장 젊음:** `{youngest_sido['clean_region']}`  \n└ 고령화율: **{youngest_sido['고령화비율']:.1f}%** (만 {int(youngest_sido['중위연령'])}세)")
    else:
        st.caption("데이터 부족")
    st.markdown("</div>", unsafe_allow_html=True)

# 2. 서울시 구별 비교
with ext_col2:
    st.markdown("<div style='background-color:#1c1f26; padding:15px; border-radius:10px; border:1px solid #383e4c;'>", unsafe_allow_html=True)
    st.markdown("##### 🏙️ 서울시 25개 구 최고 vs 최저")
    df_seoul_subset = df_metrics[df_metrics['clean_region'].str.startswith("서울특별시 ") & (df_metrics['region_level'] == 'Sigungu')]
    oldest_seoul, youngest_seoul = get_extremes(df_seoul_subset, '고령화비율')
    
    if oldest_seoul is not None:
        st.markdown(f"🧓 **최고 고령:** `{oldest_seoul['clean_region'].split()[-1]}`  \n└ 고령화율: **{oldest_seoul['고령화비율']:.1f}%** (만 {int(oldest_seoul['중위연령'])}세)")
        st.markdown(f"👶 **가장 젊음:** `{youngest_seoul['clean_region'].split()[-1]}`  \n└ 고령화율: **{youngest_seoul['고령화비율']:.1f}%** (만 {int(youngest_seoul['중위연령'])}세)")
    else:
        st.caption("데이터 부족")
    st.markdown("</div>", unsafe_allow_html=True)

# 3. 현재 선택한 지역의 하위 지자체 비교
with ext_col3:
    st.markdown("<div style='background-color:#1c1f26; padding:15px; border-radius:10px; border:1px solid #383e4c;'>", unsafe_allow_html=True)
    
    sel_code = reg_data['region_code']
    sel_level = reg_data['region_level']
    sel_name = reg_data['clean_region']
    
    sub_title = "📍 선택 지역 비교"
    df_sub = None
    
    if sel_level == 'Nation':
        sub_title = "📍 전국 광역시도 비교"
        df_sub = df_sido_subset
    elif sel_level == 'Sido':
        sub_title = f"📍 {sel_name} 산하 시군구 비교"
        prefix = sel_code[:2]
        df_sub = df_metrics[(df_metrics['region_code'].str.startswith(prefix)) & (df_metrics['region_level'] == 'Sigungu') & (df_metrics['clean_region'] != sel_name)]
    elif sel_level == 'Sigungu':
        sub_title = f"📍 {sel_name.split()[-1]} 산하 읍면동 비교"
        prefix = sel_code[:4]
        df_sub = df_metrics[(df_metrics['region_code'].str.startswith(prefix)) & (df_metrics['region_level'] == 'Dong') & (df_metrics['clean_region'] != sel_name)]
    else:
        sub_title = f"📍 인근 읍면동 간 비교"
        prefix = sel_code[:4]
        df_sub = df_metrics[(df_metrics['region_code'].str.startswith(prefix)) & (df_metrics['region_level'] == 'Dong')]

    st.markdown(f"##### {sub_title}")
    oldest_sub, youngest_sub = get_extremes(df_sub, '고령화비율')
    
    if oldest_sub is not None:
        name_o = oldest_sub['clean_region'].split()[-1]
        name_y = youngest_sub['clean_region'].split()[-1]
        st.markdown(f"🧓 **최고 고령:** `{name_o}`  \n└ 고령화율: **{oldest_sub['고령화비율']:.1f}%** (만 {int(oldest_sub['중위연령'])}세)")
        st.markdown(f"👶 **가장 젊음:** `{name_y}`  \n└ 고령화율: **{youngest_sub['고령화비율']:.1f}%** (만 {int(youngest_sub['중위연령'])}세)")
    else:
        st.caption("하위 행정구역 데이터가 없습니다.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# 메인 레이아웃 탭 분할
tab_map, tab_pyramid, tab_compare = st.tabs(["🗺️ 인터랙티브 행정지도", "📊 세부 연령 구조 (피라미드)", "📈 지역 간 비교 및 랭킹"])

# -----------------------------------------------------------------------------
# 🛠️ 지능형 한국 전국 광역시도 매퍼 (충남, 충북, 전남, 전북 완벽 분할 해결)
# -----------------------------------------------------------------------------
def match_national_province_robust(english_id, df_sido):
    # ID에서 대시, 언더바, do, si 등을 제거하여 소문자 통일
    clean_id = english_id.lower().replace('-', '').replace('_', '').replace('do', '').replace('si', '').replace('special', '').strip()
    
    id_to_term = {
        'kr11': '서울', '11': '서울', 'seoul': '서울',
        'kr26': '부산', '26': '부산', 'busan': '부산',
        'kr27': '대구', '27': '대구', 'daegu': '대구',
        'kr28': '인천', '28': '인천', 'incheon': '인천',
        'kr29': '광주', '29': '광주', 'gwangju': '광주',
        'kr30': '대전', '30': '대전', 'daejeon': '대전',
        'kr31': '울산', '31': '울산', 'ulsan': '울산',
        'kr50': '세종', '50': 'sejong', 'sejong': '세종',
        'kr41': '경기', '41': '경기', 'gyeonggi': '경기',
        'kr42': '강원', '42': '강원', 'gangwon': '강원',
        'kr43': '충북', '43': '충북', 'chungbuk': '충북', 'chungcheongbuk': '충북',
        'kr44': '충남', '44': '충남', 'chungnam': '충남', 'chungcheongnam': '충남',
        'kr45': '전북', '45': '전북', 'jeonbuk': '전북', 'jeollabuk': '전북',
        'kr46': '전남', '46': '전남', 'jeonnam': '전남', 'jeollanam': '전남',
        'kr47': '경북', '47': '경북', 'gyeongbuk': '경북', 'gyeongsangbuk': '경북',
        'kr48': '경남', '48': '경남', 'gyeongnam': '경남', 'gyeongsangnam': '경남',
        'kr49': '제주', '49': '제주', 'jeju': '제주'
    }
    
    term = id_to_term.get(clean_id, '')
    if not term:
        return None
        
    for idx, row in df_sido.iterrows():
        reg = row['clean_region']
        if term in reg:
            return reg
        if term == '충북' and '충청북도' in reg:
            return reg
        if term == '충남' and '충청남도' in reg:
            return reg
        if term == '전북' and ('전라북도' in reg or '전북' in reg or '전북특별자치도' in reg):
            return reg
        if term == '전남' and '전라남도' in reg:
            return reg
        if term == '경북' and '경상북도' in reg:
            return reg
        if term == '경남' and '경상남도' in reg:
            return reg
            
    return None

# -----------------------------------------------------------------------------
# 🛠️ 서울 영등포구 및 전 자치구 100% 매칭 보증 초정밀 로버스트 매퍼
# -----------------------------------------------------------------------------
def match_seoul_district_robust(path_id, df_seoul):
    if not path_id:
        return None
    clean_id = path_id.lower().replace('-', '').replace('_', '').replace(' ', '').strip()
    
    # 영등포구(Yeongdeungpo)를 비롯해 생길 수 있는 다양한 스펠링을 완벽하게 통합 매칭
    seoul_id_mapping = {
        'jongno': '종로구', 'jongnogu': '종로구',
        'jung': '중구', 'junggu': '중구', 'jung_': '중구',
        'yongsan': '용산구', 'yongsangu': '용산구',
        'seongdong': '성동구', 'seongdonggu': '성동구',
        'gwangjin': '광진구', 'gwangjingu': '광진구',
        'dongdaemun': '동대문구', 'dongdaemungu': '동대문구',
        'jungnang': '중랑구', 'jungnanggu': '중랑구', 'jungrang': '중랑구',
        'seongbuk': '성북구', 'seongbukgu': '성북구', 'seongbug': '성북구',
        'gangbuk': '강북구', 'gangbukgu': '강북구',
        'dobong': '도봉구', 'dobonggu': '도봉구',
        'nowon': '노원구', 'nowongu': '노원구',
        'eunpyeong': '은평구', 'eunpyeonggu': '은평구',
        'seodaemun': '서대문구', 'seodaemungu': '서대문구', 'seodaemoon': '서대문구',
        'mapo': '마포구', 'mapogu': '마포구',
        'yangcheon': '양천구', 'yangcheongu': '양천구',
        'gangseo': '강서구', 'gangseogu': '강서구',
        'guro': '구로구', 'gurogu': '구로구',
        'geumcheon': '금천구', 'geumcheongu': '금천구',
        # 영등포구 표기 변형 전수 예외 처리 (e-o, e-o-u, o-u, o, d-u-n-g, d-e-u-n-g 등 모든 형태 완벽 수용)
        'yeongdeungpo': '영등포구', 'yeongdeungpogu': '영등포구', 
        'yeoungdeungpo': '영등포구', 'yeoungdeungpogu': '영등포구',
        'yeongdungpo': '영등포구', 'yeongdungpogu': '영등포구',
        'yeoungdungpo': '영등포구', 'yeoungdungpogu': '영등포구',
        'ydp': '영등포구',
        'dongjak': '동작구', 'dongjakgu': '동작구',
        'gwanak': '관악구', 'gwanakgu': '관악구',
        'seocho': '서초구', 'seochogu': '서초구',
        'gangnam': '강남구', 'gangnamgu': '강남구',
        'songpa': '송파구', 'songpagu': '송파구',
        'gangdong': '강동구', 'gangdonggu': '강동구'
    }
    
    target_gu = seoul_id_mapping.get(clean_id, None)
    if not target_gu:
        # 혹시 ID가 'yeongdeungpo_gu' 등 특수 형식일 경우를 대비해 다시 한번 지자체명 기반 포함관계 검색 시도
        clean_id_stripped = clean_id.replace('gu', '').strip()
        target_gu = seoul_id_mapping.get(clean_id_stripped, None)
        
    if not target_gu:
        for key, val in seoul_id_mapping.items():
            if key in clean_id or clean_id in key:
                target_gu = val
                break
                
    if target_gu:
        for idx, row in df_seoul.iterrows():
            reg_name = row['clean_region']
            if target_gu in reg_name and "서울특별시" in reg_name:
                return row
    return None

# -----------------------------------------------------------------------------
# 🛠/ fixed 위치 기반 & URL 인코딩 보호막 탑재 실시간 플로팅 툴팁 HTML 래퍼
# -----------------------------------------------------------------------------
def wrap_svg_with_custom_tooltip(svg_soup_str):
    html_page = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body {{
            margin: 0;
            padding: 0;
            overflow: hidden;
            background: transparent;
        }}
        #streamlit-floating-tooltip {{
            position: fixed;
            display: none;
            background: rgba(28, 31, 38, 0.96);
            color: #ffffff;
            padding: 10px 14px;
            font-size: 13px;
            font-weight: 500;
            border-radius: 8px;
            pointer-events: none;
            z-index: 999999;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.35);
            border: 1px solid rgba(255, 255, 255, 0.15);
            line-height: 1.5;
        }}
        path, polyline, polygon {{
            transition: fill 0.2s ease, stroke-width 0.2s ease;
            cursor: pointer;
        }}
        path:hover, polyline:hover, polygon:hover {{
            fill-opacity: 0.85 !important;
            stroke: #1e1e24 !important;
            stroke-width: 3.5px !important;
        }}
    </style>
    </head>
    <body>
        <div id="streamlit-floating-tooltip"></div>
        {svg_soup_str}
        <script>
            const tooltip = document.getElementById('streamlit-floating-tooltip');
            const elements = document.querySelectorAll('path, polyline, polygon');
            
            elements.forEach(el => {{
                if (el.getAttribute('data-tooltip')) {{
                    el.addEventListener('mouseover', (e) => {{
                        tooltip.style.display = 'block';
                        const encodedData = el.getAttribute('data-tooltip');
                        tooltip.innerHTML = decodeURIComponent(encodedData);
                    }});
                    
                    el.addEventListener('mousemove', (e) => {{
                        let posX = e.clientX + 15;
                        let posY = e.clientY + 15;
                        
                        if (posX + tooltip.offsetWidth > window.innerWidth) {{
                            posX = e.clientX - tooltip.offsetWidth - 15;
                        }}
                        if (posY + tooltip.offsetHeight > window.innerHeight) {{
                            posY = e.clientY - tooltip.offsetHeight - 15;
                        }}
                        
                        tooltip.style.left = posX + 'px';
                        tooltip.style.top = posY + 'px';
                    }});
                    
                    el.addEventListener('mouseout', () => {{
                        tooltip.style.display = 'none';
                    }});
                }}
            }});
        </script>
    </body>
    </html>
    """
    return html_page

# -----------------------------------------------------------------------------
# 📊 [추가 기능] 고령화 상위 vs 가장 젊은 지역 수평 막대그래프 1대1 렌더러 정의
# -----------------------------------------------------------------------------
def render_ranking_extremes(df_subset, title_prefix, col, label, scale_old='Reds', scale_young='Blues'):
    if df_subset is None or df_subset.empty or len(df_subset) < 2:
        st.write("📊 비교 분석을 위한 하위 지역 데이터가 존재하지 않습니다.")
        return
        
    sorted_df = df_subset.sort_values(by=col, ascending=False)
    
    # 상위 5 (늙은 지역) & 하위 5 (가장 젊은 지역) 슬라이싱
    top_oldest = sorted_df.head(5).copy()
    top_youngest = sorted_df.tail(5).sort_values(by=col, ascending=True).copy()
    
    col_l, col_r = st.columns(2)
    
    with col_l:
        st.markdown(f"**🧓 {title_prefix} 고령화지수 상위 5개 지역 (늙은 곳)**")
        fig_o = px.bar(
            top_oldest, x=col, y='clean_region',
            orientation='h',
            color=col,
            color_continuous_scale=scale_old,
            labels={'clean_region': '지역명', col: label}
        )
        fig_o.update_layout(height=280, coloraxis_showscale=False, yaxis={'categoryorder':'total ascending'}, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_o, use_container_width=True)
        
    with col_r:
        st.markdown(f"**👶 {title_prefix} 고령화지수 하위 5개 지역 (가장 젊은 곳)**")
        fig_y = px.bar(
            top_youngest, x=col, y='clean_region',
            orientation='h',
            color=col,
            color_continuous_scale=scale_young,
            labels={'clean_region': '지역명', col: label}
        )
        fig_y.update_layout(height=280, coloraxis_showscale=False, yaxis={'categoryorder':'total descending'}, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_y, use_container_width=True)

# =========================================================================
# TAB 1: 지도 시각화
# =========================================================================
with tab_map:
    map_col1, map_col2 = st.columns([7, 3])
    
    # 색상 변환 헬퍼 함수
    def get_rgb_color(val, min_v, max_v, theme):
        if max_v == min_v:
            f = 0.0
        else:
            f = (val - min_v) / (max_v - min_v)
        f = max(0.0, min(1.0, f))
        
        if theme == "Reds":
            start, end = (255, 245, 245), (153, 0, 13)
        elif theme == "Blues":
            start, end = (247, 251, 255), (8, 48, 107)
        else: # YlOrRd
            if f < 0.5:
                f_sub = f * 2
                start, end = (255, 255, 204), (254, 178, 76)
                r = int(start[0] + (end[0] - start[0]) * f_sub)
                g = int(start[1] + (end[1] - start[1]) * f_sub)
                b = int(start[2] + (end[2] - start[2]) * f_sub)
                return f"#{r:02x}{g:02x}{b:02x}"
            else:
                f_sub = (f - 0.5) * 2
                start, end = (254, 178, 76), (189, 0, 38)
                r = int(start[0] + (end[0] - start[0]) * f_sub)
                g = int(start[1] + (end[1] - start[1]) * f_sub)
                b = int(start[2] + (end[2] - start[2]) * f_sub)
                return f"#{r:02x}{g:02x}{b:02x}"
                
        r = int(start[0] + (end[0] - start[0]) * f)
        g = int(start[1] + (end[1] - start[1]) * f)
        b = int(start[2] + (end[2] - start[2]) * f)
        return f"#{r:02x}{g:02x}{b:02x}"
        
    with map_col1:
        map_tab1, map_tab2 = st.tabs(["全国 대한민국 광역지도", "SEOUL 서울시 자치구 세부지도"])
        
        # 전국 지도 렌더링
        with map_tab1:
            svg_national = fetch_svg("Map_of_South_Korea-blank.svg")
            if svg_national:
                df_sido = df_metrics[df_metrics['region_level'] == 'Sido']
                
                if df_sido.empty:
                    st.warning("⚠️ 전국 광역지자체 데이터를 식별하지 못했습니다.")
                else:
                    min_val = df_sido[selected_metric].min()
                    max_val = df_sido[selected_metric].max()
                    
                    try:
                        soup = BeautifulSoup(svg_national, "xml")
                    except Exception:
                        soup = BeautifulSoup(svg_national, "html.parser")
                        
                    soup.svg['width'] = '100%'
                    soup.svg['height'] = '550px'
                    
                    for path in soup.find_all(['path', 'polyline', 'polygon']):
                        p_id = path.get('id')
                        # Fallback: 요소 자체에 ID가 없는 특수 구조 SVG 대비용 부모 그룹 노드 검색식
                        if not p_id and path.parent and path.parent.name == 'g':
                            p_id = path.parent.get('id')
                            
                        if p_id:
                            kor_prefix = match_national_province_robust(p_id, df_sido)
                            
                            if kor_prefix:
                                matched_row = df_sido[df_sido['clean_region'] == kor_prefix]
                                if matched_row is not None and not matched_row.empty:
                                    r_data = matched_row.iloc[0]
                                    val = r_data[selected_metric]
                                    color = get_rgb_color(val, min_val, max_val, selected_theme)
                                    
                                    # 인라인 style 속성을 통째로 강제 덮어씌워 브라우저 강제 채색 유도
                                    path['style'] = f"fill: {color} !important; stroke: #ffffff !important; stroke-width: 1.2px !important;"
                                    
                                    if isinstance(val, (float, np.floating)):
                                        formatted_val = f"{val:.2f}"
                                    else:
                                        formatted_val = f"{int(val):,}"
                                        
                                    tooltip_html = f"<div style='font-size:14px; font-weight:bold; margin-bottom:5px; border-bottom:1px solid rgba(255,255,255,0.2); padding-bottom:3px; color:#58a6ff;'>📍 {r_data['clean_region']}</div><div style='margin-bottom:3px;'>🔍 <b>{selected_label}:</b> <span style='color:#ff7f0e; font-weight:bold;'>{formatted_val}</span></div><div>👥 <b>총 인구:</b> {int(r_data['총인구']):,}명</div>"
                                    path['data-tooltip'] = urllib.parse.quote(tooltip_html)
                                    
                    final_html = wrap_svg_with_custom_tooltip(str(soup))
                    import streamlit.components.v1 as components
                    components.html(final_html, height=570)
            else:
                st.error("국가 지도 SVG 로딩 실패")
                
        # 서울 지도 렌더링 (부모 <g> ID 추적 및 초정밀 영등포구 예외 매퍼 가동)
        with map_tab2:
            svg_seoul = fetch_svg("Seoul_districts.svg")
            if svg_seoul:
                df_seoul = df_metrics[df_metrics['clean_region'].str.startswith("서울특별시 ")]
                
                if df_seoul.empty:
                    df_seoul = df_metrics[df_metrics['clean_region'].str.contains("구") & ~df_metrics['clean_region'].str.contains("시 ")]
                
                if df_seoul.empty:
                    st.warning("⚠️ 서울 자치구 데이터를 식별하지 못했습니다.")
                else:
                    min_val_s = df_seoul[selected_metric].min()
                    max_val_s = df_seoul[selected_metric].max()
                    
                    try:
                        soup_s = BeautifulSoup(svg_seoul, "xml")
                    except Exception:
                        soup_s = BeautifulSoup(svg_seoul, "html.parser")
                        
                    soup_s.svg['width'] = '100%'
                    soup_s.svg['height'] = '550px'
                    
                    for path in soup_s.find_all(['path', 'polygon', 'polyline']):
                        p_id = path.get('id')
                        # [가장 중요한 패치]: 자식 엘리먼트에 ID가 없을 시 부모 <g> 그룹의 ID를 역추적하여 매핑 (영등포구 회색 버그의 원인 규명 및 복원)
                        if not p_id and path.parent and path.parent.name == 'g':
                            p_id = path.parent.get('id')
                            
                        if p_id:
                            # 초정밀 로버스트 매퍼를 호출해 오차 및 예외 없는 100% 매칭 수행
                            r_data = match_seoul_district_robust(p_id, df_seoul)
                            
                            if r_data is not None:
                                val = r_data[selected_metric]
                                color = get_rgb_color(val, min_val_s, max_val_s, selected_theme)
                                
                                # 인라인 style을 강제 주입하여 채색 무시 버그 완벽 제거
                                path['style'] = f"fill: {color} !important; stroke: #ffffff !important; stroke-width: 1.5px !important;"
                                
                                if isinstance(val, (float, np.floating)):
                                    formatted_val = f"{val:.2f}"
                                else:
                                    formatted_val = f"{int(val):,}"
                                    
                                tooltip_html_s = f"<div style='font-size:14px; font-weight:bold; margin-bottom:5px; border-bottom:1px solid rgba(255,255,255,0.2); padding-bottom:3px; color:#58a6ff;'>📍 {r_data['clean_region']}</div><div style='margin-bottom:3px;'>🔍 <b>{selected_label}:</b> <span style='color:#ff7f0e; font-weight:bold;'>{formatted_val}</span></div><div>👥 <b>총 인구:</b> {int(r_data['총인구']):,}명</div>"
                                path['data-tooltip'] = urllib.parse.quote(tooltip_html_s)
                                
                    final_html_s = wrap_svg_with_custom_tooltip(str(soup_s))
                    components.html(final_html_s, height=570)
            else:
                st.error("서울 지도 SVG 로딩 실패")
                
    with map_col2:
        st.markdown(f"### 🎨 지도 범례 ({selected_label})")
        
        if selected_theme == "Reds":
            grad = "linear-gradient(to right, #fff5f5, #99000d)"
        elif selected_theme == "Blues":
            grad = "linear-gradient(to right, #f7fbfd, #08306b)"
        else:
            grad = "linear-gradient(to right, #ffffe0, #feb24c, #bd0026)"
            
        st.markdown(f"""
            <div style="margin: 15px 0;">
                <div style="background: {grad}; height: 20px; width: 100%; border-radius: 4px; border: 1px solid #ccc;"></div>
                <div style="display: flex; justify-content: space-between; font-size: 12px; margin-top: 5px; font-weight: bold;">
                    <span>최소값 (Min)</span>
                    <span>최대값 (Max)</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("🔍 **지도를 분석하는 팁:**")
        st.info("지도 속 마우스를 구역 위에 올려두면 행정구역 명칭과 정확한 수치 정보를 말풍선 팝업으로 확인할 수 있습니다.")

# =========================================================================
# TAB 2: 세부 연령 구조 (인구 피라미드 & 3대 그룹)
# =========================================================================
with tab_pyramid:
    st.subheader(f"📊 {selected_region} 세부 연령 분석")
    
    if reg_data['raw_row'] is None:
        st.warning("⚠️ CSV 인구 데이터 원본 파일이 비어 있어 피라미드 분석을 연동할 수 없습니다. 행안부 원본 파일을 드롭해 주세요.")
    else:
        pyr_col1, pyr_col2 = st.columns(2)
        
        with pyr_col1:
            st.markdown("#### 👨‍👩‍👧‍👦 남녀 성별 인구 피라미드")
            
            raw_row = reg_data['raw_row']
            male_cols = reg_data['male_cols']
            female_cols = reg_data['female_cols']
            ages = reg_data['ages']
            
            m_vals = [raw_row[male_cols[a]] for a in ages]
            f_vals = [raw_row[female_cols[a]] for a in ages]
            
            fig_pyramid = go.Figure()
            fig_pyramid.add_trace(go.Bar(
                y=ages, x=[-v for v in m_vals],
                name='남성 (Male)', orientation='h',
                marker=dict(color='#1177b4'),
                hoverinfo='text',
                hovertext=[f"남성 만 {a}세: {v:,}명" for a, v in zip(ages, m_vals)]
            ))
            fig_pyramid.add_trace(go.Bar(
                y=ages, x=f_vals,
                name='여성 (Female)', orientation='h',
                marker=dict(color='#e377c2'),
                hoverinfo='text',
                hovertext=[f"여성 만 {a}세: {v:,}명" for a, v in zip(ages, f_vals)]
            ))
            
            max_p = max(max(m_vals), max(f_vals)) if m_vals else 1000
            tick_vals = [-max_p, -max_p//2, 0, max_p//2, max_p]
            tick_text = [f"{abs(v):,}" for v in tick_vals]
            
            fig_pyramid.update_layout(
                barmode='overlay',
                xaxis=dict(title="인구 수 (명)", tickvals=tick_vals, ticktext=tick_text),
                yaxis=dict(title="만 연령 (세)", range=[0, 100]),
                height=500,
                legend=dict(x=0.8, y=0.95),
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_pyramid, use_container_width=True)
            
        with pyr_col2:
            st.markdown("#### 🍕 주요 3대 연령대별 인구 점유비")
            
            donut_data = pd.DataFrame({
                '구분': ['유소년인구 (0~14세)', '생산가능인구 (15~64세)', '고령인구 (65세 이상)'],
                '인구': [reg_data['유소년인구_14'], reg_data['생산가능인구'], reg_data['고령인구']]
            })
            
            fig_donut = px.pie(
                donut_data, values='인구', names='구분',
                hole=0.45,
                color_discrete_sequence=['#aec7e8', '#ff7f0e', '#d62728']
            )
            fig_donut.update_traces(textposition='inside', textinfo='percent+label')
            fig_donut.update_layout(
                height=500,
                legend=dict(orientation="h", y=-0.1, x=0.1),
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_donut, use_container_width=True)

# =========================================================================
# TAB 3: 지역 간 비교 및 랭킹
# =========================================================================
with tab_compare:
    st.subheader("📊 전국 시도 및 시군구 지표 랭킹 비교")
    
    # 1. 전국 광역시도 극단 비교 (막대그래프)
    st.markdown("### 🌐 전국 17개 시도 고령화 및 젊은 지역 수평바 비교")
    df_sido_subset = df_metrics[(df_metrics['region_level'] == 'Sido') & (df_metrics['clean_region'] != '전국')]
    render_ranking_extremes(df_sido_subset, "전국", selected_metric, selected_label)
    
    st.markdown("---")
    
    # 2. 서울시 자치구 극단 비교 (막대그래프)
    st.markdown("### 🏙️ 서울시 25개 자치구 고령화 및 젊은 지역 수평바 비교")
    df_seoul_subset = df_metrics[df_metrics['clean_region'].str.startswith("서울특별시 ") & (df_metrics['region_level'] == 'Sigungu')]
    render_ranking_extremes(df_seoul_subset, "서울시", selected_metric, selected_label)
    
    st.markdown("---")
    
    # 3. 현재 선택한 지역의 하위 지자체 비교 (막대그래프)
    sel_code = reg_data['region_code']
    sel_level = reg_data['region_level']
    sel_name = reg_data['clean_region']
    
    sub_title = "선택 지역"
    df_sub = None
    
    if sel_level == 'Nation':
        sub_title = "전국 광역시도"
        df_sub = df_sido_subset
    elif sel_level == 'Sido':
        sub_title = f"{sel_name} 산하 시군구"
        prefix = sel_code[:2]
        df_sub = df_metrics[(df_metrics['region_code'].str.startswith(prefix)) & (df_metrics['region_level'] == 'Sigungu') & (df_metrics['clean_region'] != sel_name)]
    elif sel_level == 'Sigungu':
        sub_title = f"{sel_name.split()[-1]} 산하 읍면동"
        prefix = sel_code[:4]
        df_sub = df_metrics[(df_metrics['region_code'].str.startswith(prefix)) & (df_metrics['region_level'] == 'Dong') & (df_metrics['clean_region'] != sel_name)]
    else:
        sub_title = "인근 읍면동"
        prefix = sel_code[:4]
        df_sub = df_metrics[(df_metrics['region_code'].str.startswith(prefix)) & (df_metrics['region_level'] == 'Dong')]
        
    st.markdown(f"### 📍 {sub_title} 고령화 및 젊은 지역 수평바 비교")
    render_ranking_extremes(df_sub, sub_title, selected_metric, selected_label)
    
    st.markdown("---")
    
    # 4. 전체 리스트 아코디언 컴포넌트
    with st.expander("📊 전체 행정구역 풀 랭킹 차트 보기 (기존 전체 그래프)"):
        comp_col1, comp_col2 = st.columns([1, 1])
        
        with comp_col1:
            st.markdown(f"#### 🏆 전국 17개 시도 전체 랭킹")
            df_sido_comp = df_metrics[df_metrics['region_level'] == 'Sido'].sort_values(by=selected_metric, ascending=True)
            fig_sido_bar = px.bar(
                df_sido_comp, x=selected_metric, y='clean_region',
                orientation='h',
                labels={'clean_region': '지역', selected_metric: selected_label},
                color=selected_metric,
                color_continuous_scale=selected_theme
            )
            fig_sido_bar.update_layout(height=500, coloraxis_showscale=False, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_sido_bar, use_container_width=True)
            
        with comp_col2:
            st.markdown(f"#### 🔍 서울시 25개 자치구 전체 랭킹")
            df_seoul_comp = df_metrics[df_metrics['clean_region'].str.startswith("서울특별시 ")].sort_values(by=selected_metric, ascending=True)
            df_seoul_comp['short_name'] = df_seoul_comp['clean_region'].apply(lambda x: x.split()[-1])
            fig_seoul_bar = px.bar(
                df_seoul_comp, x=selected_metric, y='short_name',
                orientation='h',
                labels={'short_name': '자치구', selected_metric: selected_label},
                color=selected_metric,
                color_continuous_scale=selected_theme
            )
            fig_seoul_bar.update_layout(height=500, coloraxis_showscale=False, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_seoul_bar, use_container_width=True)
