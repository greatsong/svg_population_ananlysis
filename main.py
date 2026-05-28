import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import json
import re
from bs4 import BeautifulSoup

# 페이지 설정
st.set_page_config(
    page_title="대한민국 지역별 인구구조 분석 대시보드",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    
    # 행정구역 코드 체계를 통한 지역 레벨 분류 (코드 체계 부재 시 예외처리 포함)
    def classify_level(row):
        code = row['region_code']
        name = row['clean_region']
        if code != '0000000000':
            if code == '0000000000' or name == '전국':
                return 'Nation'
            elif code.endswith('00000000'):
                return 'Sido'
            elif code.endswith('00000'):
                return 'Sigungu'
            else:
                return 'Dong'
        else:
            # 코드가 없는 구버전 CSV 등일 때 명칭 패턴 분석
            if name in ['전국', '합계']:
                return 'Nation'
            sido_list = ["서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시", "대전광역시", "울산광역시", "세종특별자치시", "경기도", "강원도", "충청북도", "충청남도", "전라북도", "전라남도", "경상북도", "경상남도", "제주특별자치도", "강원특별자치도", "전북특별자치도"]
            if name in sido_list or name.endswith('도') or name.endswith('특별시') or name.endswith('광역시') or name.endswith('특별자치시') or name.endswith('특별자치도'):
                return 'Sido'
            elif name.endswith('구') or name.endswith('시') or name.endswith('군'):
                return 'Sigungu'
            else:
                return 'Dong'
            
    df['region_level'] = df.apply(classify_level, axis=1)
    
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
    regions = ["서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시", "대전광역시", "울산광역시", "세종특별자치시", "경기도", "강원도", "충청북도", "충청남도", "전라북도", "전라남도", "경상북도", "경상남도", "제주특별자치도"]
    seoul_districts = ["종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구", "성북구", "강북구", "도봉구", "노원구", "은평구", "서대문구", "마포구", "양천구", "강서구", "구로구", "금천구", "영등포구", "동작구", "관악구", "서초구", "강남구", "송파구", "강동구"]
    
    data = []
    import random
    random.seed(42)
    for r in regions:
        tot = random.randint(1000000, 13000000) if "경기" in r else random.randint(300000, 3400000)
        data.append({
            'clean_region': r, 'region_code': '1100000000', 'region_level': 'Sido',
            '총인구': tot, '남성인구': int(tot*0.49), '여성인구': int(tot*0.51),
            '유소년인구_14': int(tot*0.11), '청소년인구_18': int(tot*0.15),
            '생산가능인구': int(tot*0.70), '고령인구': int(tot*0.19),
            '고령화비율': random.uniform(14.0, 25.0), '청소년비율': random.uniform(10.0, 18.0),
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
# 4. 분석 엔진 및 메인 대시보드 가동
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

st.markdown("---")

# 메인 레이아웃 탭 분할
tab_map, tab_pyramid, tab_compare = st.tabs(["🗺️ 인터랙티브 행정지도", "📊 세부 연령 구조 (피라미드)", "📈 지역 간 비교 및 랭킹"])

# -----------------------------------------------------------------------------
# 🛠️ 전국 시도 및 서울시 자치구 ID 완벽 매핑용 전수조사 다변 맵 사전 정의
# -----------------------------------------------------------------------------
SIDO_ALIASES = {
    '서울특별시': ['seoul', 'kr-11', '11'],
    '부산광역시': ['busan', 'kr-26', '26'],
    '대구광역시': ['daegu', 'kr-27', '27'],
    '인천광역시': ['incheon', 'kr-28', '28'],
    '광주광역시': ['gwangju', 'kr-29', '29'],
    '대전광역시': ['daejeon', 'kr-30', '30'],
    '울산광역시': ['ulsan', 'kr-31', '31'],
    '세종특별자치시': ['sejong', 'kr-50', '50'],
    '경기도': ['gyeonggi', 'kr-41', '41'],
    '강원': ['gangwon', 'kr-42', '42'], # 강원도, 강원특별자치도 모두 대변
    '충청북도': ['chungbuk', 'kr-43', '43'],
    '충청남도': ['chungnam', 'kr-44', '44'],
    '전라북도': ['jeonbuk', 'kr-45', '45'], # 전라북도, 전북특별자치도 대변
    '전라남도': ['jeonnam', 'kr-46', '46'],
    '경상북도': ['gyeongbuk', 'kr-47', '47'],
    '경상남도': ['gyeongnam', 'kr-48', '48'],
    '제주특별자치도': ['jeju', 'kr-49', '49']
}

SEOUL_DISTRICTS_ALIASES = {
    '종로구': ['jongno', 'jongno-gu', 'jongno_gu'],
    '중구': ['jung', 'jung-gu', 'jung_gu', 'jung_'],
    '용산구': ['yongsan', 'yongsan-gu', 'yongsan_gu'],
    '성동구': ['seongdong', 'seongdong-gu', 'seongdong_gu'],
    '광진구': ['gwangjin', 'gwangjin-gu', 'gwangjin_gu'],
    '동대문구': ['dongdaemun', 'dongdaemun-gu', 'dongdaemun_gu'],
    '중랑구': ['jungnang', 'jungnang-gu', 'jungnang_gu'],
    '성북구': ['seongbuk', 'seongbuk-gu', 'seongbuk_gu'],
    '강북구': ['gangbuk', 'gangbuk-gu', 'gangbuk_gu'],
    '도봉구': ['dobong', 'dobong-gu', 'dobong_gu'],
    '노원구': ['nowon', 'nowon-gu', 'nowon_gu'],
    '은평구': ['eunpyeong', 'eunpyeong-gu', 'eunpyeong_gu'],
    '서대문구': ['seodaemun', 'seodaemun-gu', 'seodaemun_gu'],
    '마포구': ['mapo', 'mapo-gu', 'mapo_gu'],
    '양천구': ['yangcheon', 'yangcheon-gu', 'yangcheon_gu'],
    '강서구': ['gangseo', 'gangseo-gu', 'gangseo_gu'],
    '구로구': ['guro', 'guro-gu', 'guro_gu'],
    '금천구': ['geumcheon', 'geumcheon-gu', 'geumcheon_gu'],
    '영등포구': ['yeongdeungpo', 'yeongdeungpo-gu', 'yeongdeungpo_gu'],
    '동작구': ['dongjak', 'dongjak-gu', 'dongjak_gu'],
    '관악구': ['gwanak', 'gwanak-gu', 'gwanak_gu'],
    '서초구': ['seocho', 'seocho-gu', 'seocho_gu'],
    '강남구': ['gangnam', 'gangnam-gu', 'gangnam_gu'],
    '송파구': ['songpa', 'songpa-gu', 'songpa_gu'],
    '강동구': ['gangdong', 'gangdong-gu', 'gangdong_gu']
}

# -----------------------------------------------------------------------------
# 🛠️ 지능형 한국 전국 광역 매퍼 정의 (충청, 전라 등 광역도 단위 완벽 판정)
# -----------------------------------------------------------------------------
def find_sido_row(alias_key, df_sido):
    for idx, row in df_sido.iterrows():
        reg_name = row['clean_region']
        
        # 1. 완전 일치
        if alias_key == reg_name:
            return row
            
        # 2. 포함 관계 일치
        if (alias_key in reg_name) or (reg_name in alias_key):
            return row
            
        # 3. 강원/전북 등 특별자치도 명칭 전수 검색 보완
        if '강원' in alias_key and '강원' in reg_name:
            return row
        if '전북' in alias_key and ('전북' in reg_name or '전라북도' in reg_name):
            return row
        if '전라북도' in alias_key and ('전북' in reg_name or '전라북도' in reg_name):
            return row
            
        # 4. 충북, 충남, 전북, 전남, 경북, 경남 축약어 보조 매칭
        abbrevs = {
            '충청북도': '충북', '충청남도': '충남', 
            '전라북도': '전북', '전라남도': '전남', 
            '경상북도': '경북', '경상남도': '경남'
        }
        if alias_key in abbrevs and abbrevs[alias_key] in reg_name:
            return row
            
    return None

def find_seoul_district_row(district_name, df_seoul):
    for idx, row in df_seoul.iterrows():
        reg_name = row['clean_region']
        if district_name in reg_name and "서울특별시" in reg_name:
            return row
    return None

# -----------------------------------------------------------------------------
# 5. 브라우저 렌더링용 자바스크립트 + HTML 마스터 템플릿
# -----------------------------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<style>
    body {
        margin: 0;
        padding: 0;
        overflow: hidden;
        background: transparent;
    }
    #streamlit-floating-tooltip {
        position: absolute;
        display: none;
        background: rgba(28, 31, 38, 0.98);
        color: #ffffff;
        padding: 10px 14px;
        font-size: 13px;
        font-weight: 500;
        border-radius: 8px;
        pointer-events: none;
        z-index: 10000;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.15);
        line-height: 1.5;
        transition: opacity 0.1s ease;
    }
    path, polyline, polygon, rect {
        transition: fill 0.25s ease-in-out, stroke-width 0.25s ease-in-out;
        cursor: pointer;
    }
    path:hover, polyline:hover, polygon:hover, rect:hover {
        fill-opacity: 0.8 !important;
        stroke: #1e1e24 !important;
        stroke-width: 3.5px !important;
    }
</style>
</head>
<body>
    <div id="streamlit-floating-tooltip"></div>
    __SVG_CONTENT__
    <script>
        const tooltipData = __JSON_DATA__;
        const tooltip = document.getElementById('streamlit-floating-tooltip');
        
        const elements = document.querySelectorAll('path, polyline, polygon, rect');
        elements.forEach(el => {
            const rawId = el.getAttribute('id');
            if (rawId) {
                const normId = rawId.toLowerCase().trim();
                const data = tooltipData[normId];
                if (data) {
                    // JavaScript 엔진이 인구 데이터 색상 및 테두리를 지도에 직접 드로잉
                    el.style.fill = data.color;
                    el.style.stroke = '#ffffff';
                    el.style.strokeWidth = '1.5px';
                    
                    // 마우스 무브 트래킹 및 다크 모드 말풍선 동적 생성
                    el.addEventListener('mouseover', (e) => {
                        tooltip.style.display = 'block';
                        tooltip.innerHTML = `
                            <div style="font-weight: bold; font-size: 14px; border-bottom: 1px solid rgba(255,255,255,0.2); padding-bottom: 4px; margin-bottom: 4px; color: #58a6ff;">📍 ` + data.name + `</div>
                            <div style="margin-bottom: 3px;"><b>` + data.metricLabel + `:</b> <span style="color:#ff7f0e; font-weight:bold;">` + data.value + `</span></div>
                            <div><b>총 인구:</b> ` + data.pop + `명</div>
                        `;
                    });
                    
                    el.addEventListener('mousemove', (e) => {
                        let posX = e.clientX + 15;
                        let posY = e.clientY + 15;
                        if (posX + tooltip.offsetWidth > window.innerWidth) {
                            posX = e.clientX - tooltip.offsetWidth - 15;
                        }
                        if (posY + tooltip.offsetHeight > window.innerHeight) {
                            posY = e.clientY - tooltip.offsetHeight - 15;
                        }
                        tooltip.style.left = posX + 'px';
                        tooltip.style.top = posY + 'px';
                    });
                    
                    el.addEventListener('mouseout', () => {
                        tooltip.style.display = 'none';
                    });
                }
            }
        });
    </script>
</body>
</html>
"""

# =========================================================================
# TAB 1: 지도 시각화 (동적 매핑 보강 및 JS 툴팁 적용)
# =========================================================================
with tab_map:
    st.subheader(f"🗺️ 지도로 보는 {selected_label} 공간 분포")
    
    map_col1, map_col2 = st.columns([7, 3])
    
    # 색상 추출 헬퍼 함수
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
        
        # 1. 전국 지도 렌더링
        with map_tab1:
            svg_national = fetch_svg("Map_of_South_Korea-blank.svg")
            if svg_national:
                df_sido = df_metrics[df_metrics['region_level'] == 'Sido']
                min_val = df_sido[selected_metric].min()
                max_val = df_sido[selected_metric].max()
                
                # BS4는 뷰박스 크기만 100%로 가볍게 보정
                try:
                    soup = BeautifulSoup(svg_national, "xml")
                    if soup.svg:
                        soup.svg['width'] = '100%'
                        soup.svg['height'] = '550px'
                    svg_clean = str(soup)
                except Exception:
                    svg_clean = svg_national
                
                # 파이썬 데이터 -> 자바스크립트 맵 JSON 구조화
                national_json_data = {}
                for sido_name, aliases in SIDO_ALIASES.items():
                    row = find_sido_row(sido_name, df_sido)
                    if row is not None:
                        val = row[selected_metric]
                        color = get_rgb_color(val, min_val, max_val, selected_theme)
                        
                        if isinstance(val, (float, np.floating)):
                            formatted_val = f"{val:.2f}"
                        else:
                            formatted_val = f"{int(val):,}"
                            
                        # 별칭(Alias)으로 들어오는 모든 지도상의 ID에 완벽 매칭
                        for alias in aliases:
                            national_json_data[alias.lower()] = {
                                'name': row['clean_region'],
                                'metricLabel': selected_label,
                                'value': formatted_val,
                                'pop': f"{int(row['총인구']):,}",
                                'color': color
                            }
                
                # HTML 템플릿에 SVG와 데이터맵을 주입하여 송출
                final_html = HTML_TEMPLATE.replace("__SVG_CONTENT__", svg_clean).replace("__JSON_DATA__", json.dumps(national_json_data))
                import streamlit.components.v1 as components
                components.html(final_html, height=570)
            else:
                st.error("국가 지도 SVG 데이터를 위키미디어에서 불러오지 못했습니다.")
                
        # 2. 서울 지도 렌더링
        with map_tab2:
            svg_seoul = fetch_svg("Seoul_districts.svg")
            if svg_seoul:
                df_seoul = df_metrics[df_metrics['clean_region'].str.startswith("서울특별시 ")]
                min_val_s = df_seoul[selected_metric].min()
                max_val_s = df_seoul[selected_metric].max()
                
                try:
                    soup_s = BeautifulSoup(svg_seoul, "xml")
                    if soup_s.svg:
                        soup_s.svg['width'] = '100%'
                        soup_s.svg['height'] = '550px'
                    svg_clean_s = str(soup_s)
                except Exception:
                    svg_clean_s = svg_seoul
                
                seoul_json_data = {}
                for district_name, aliases in SEOUL_DISTRICTS_ALIASES.items():
                    row = find_seoul_district_row(district_name, df_seoul)
                    if row is not None:
                        val = row[selected_metric]
                        color = get_rgb_color(val, min_val_s, max_val_s, selected_theme)
                        
                        if isinstance(val, (float, np.floating)):
                            formatted_val = f"{val:.2f}"
                        else:
                            formatted_val = f"{int(val):,}"
                            
                        for alias in aliases:
                            seoul_json_data[alias.lower()] = {
                                'name': row['clean_region'],
                                'metricLabel': selected_label,
                                'value': formatted_val,
                                'pop': f"{int(row['총인구']):,}",
                                'color': color
                            }
                
                final_html_s = HTML_TEMPLATE.replace("__SVG_CONTENT__", svg_clean_s).replace("__JSON_DATA__", json.dumps(seoul_json_data))
                components.html(final_html_s, height=570)
            else:
                st.error("서울 지도 SVG 데이터를 위키미디어에서 불러오지 못했습니다.")
                
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
        
        # ---------------------------------------------------------------------
        # 🛠️ 지도의 무결성 보장을 위한 매핑 사전 진단 판판넬 배치
        # ---------------------------------------------------------------------
        with st.expander("🛠️ 데이터 매핑 진단 정보 (Diagnostics)"):
            st.markdown("지도가 올바르게 표시되는지 매핑 로그를 확인해 보세요.")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown("**전국 광역시도 매핑 상태:**")
                diag_data = []
                for s_name, s_aliases in SIDO_ALIASES.items():
                    s_row = find_sido_row(s_name, df_sido)
                    status = "✅ 성공" if s_row is not None else "❌ 누락"
                    diag_data.append({"행정구역": s_name, "상태": status})
                st.dataframe(pd.DataFrame(diag_data), use_container_width=True, hide_index=True)
            with col_d2:
                st.markdown("**서울시 자치구 매핑 상태:**")
                diag_seoul = []
                for d_name, d_aliases in SEOUL_DISTRICTS_ALIASES.items():
                    d_row = find_seoul_district_row(d_name, df_seoul)
                    status = "✅ 성공" if d_row is not None else "❌ 누락"
                    diag_seoul.append({"자치구": d_name, "상태": status})
                st.dataframe(pd.DataFrame(diag_seoul), use_container_width=True, hide_index=True)

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
    
    comp_col1, comp_col2 = st.columns([1, 1])
    
    with comp_col1:
        st.markdown(f"#### 🏆 전국 17개 시도별 {selected_label} 순위")
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
        st.markdown(f"#### 🔍 서울시 25개 자치구별 {selected_label} 순위")
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
