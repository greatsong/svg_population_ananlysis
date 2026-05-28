import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import re
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
    # 예: "서울특별시  (1100000000)" -> 명칭: "서울특별시", 코드: "1100000000"
    df['region_code'] = df[reg_col].apply(lambda x: re.search(r'\((\d+)\)', str(x)).group(1) if re.search(r'\((\d+)\)', str(x)) else '0000000000')
    df['clean_region'] = df[reg_col].apply(lambda x: x.split('(')[0].strip() if isinstance(x, str) else str(x))
    
    # 행정구역 코드 체계를 통한 지역 레벨 분류
    # 끝자리에 8개의 0이 오면 시도(Sido), 5개의 0이 오면 시군구(Sigungu), 나머지는 읍면동(Dong)
    def classify_level(code):
        if code == '0000000000':
            return 'Nation' # 전국
        elif code.endswith('00000000'):
            return 'Sido' # 광역자치단체 (시도)
        elif code.endswith('00000'):
            return 'Sigungu' # 기초자치단체 (시군구)
        else:
            return 'Dong' # 읍면동
            
    df['region_level'] = df['region_code'].apply(classify_level)
    
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
    """
    연령별 컬럼을 추적하여 고령화비율, 청소년비율, 생산가능인구, 성비 등을 계산하여 새로운 데이터프레임으로 리턴합니다.
    """
    # 단세별 컬럼(남/여) 분류
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
                    
    # 공통 연령대 리스트 (0세 ~ 100세 이상)
    ages = sorted(list(set(male_cols.keys()).intersection(set(female_cols.keys()))))
    
    results = []
    for idx, row in df.iterrows():
        # 기본 인구수
        m_pop_by_age = {age: row[male_cols[age]] for age in ages}
        f_pop_by_age = {age: row[female_cols[age]] for age in ages}
        total_pop_by_age = {age: m_pop_by_age[age] + f_pop_by_age[age] for age in ages}
        
        m_total = sum(m_pop_by_age.values())
        f_total = sum(f_pop_by_age.values())
        total_pop = m_total + f_total
        
        if total_pop == 0:
            continue
            
        # 3대 연령대 세그먼트 (UN 및 국내 법령 기준)
        # 1. 유소년 인구 (0~14세)
        youth_14 = sum([total_pop_by_age[a] for a in ages if a <= 14])
        # 2. 어린이 및 청소년 인구 (0~18세 - 아동복지법/청소년기본법 준용)
        youth_18 = sum([total_pop_by_age[a] for a in ages if a <= 18])
        # 3. 생산가능인구 (15~64세)
        working_age = sum([total_pop_by_age[a] for a in ages if 15 <= a <= 64])
        # 4. 고령 인구 (65세 이상)
        elderly = sum([total_pop_by_age[a] for a in ages if a >= 65])
        
        # 세부 비율 계산
        aging_ratio = (elderly / total_pop) * 100          # 고령인구 비율
        youth_ratio = (youth_18 / total_pop) * 100         # 청소년(0-18세) 비율
        working_ratio = (working_age / total_pop) * 100    # 생산가능인구 비율
        sex_ratio = (m_total / f_total) * 100 if f_total > 0 else 100  # 성비 (여성 100명당 남성 수)
        
        # 노령화지수 = (고령인구 / 유소년인구(0~14세)) * 100
        aging_index = (elderly / youth_14) * 100 if youth_14 > 0 else 0
        
        # 중위 연령 계산
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
            'raw_row': row, # 피라미드 그릴 때 사용하기 위해 원본 행 참조 저장
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
    st.sidebar.warning("⚠️ CSV 파일이 없습니다. 사이드바에서 파일을 업로드해 주세요.")

# -----------------------------------------------------------------------------
# 4. 분석 엔진 및 메인 대시보드
# -----------------------------------------------------------------------------
if raw_data is not None:
    # 데이터 모델 연산 가동
    df_metrics = calculate_demographics(raw_data)
    
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
        stage = "🔴 초고령사회 (Super-aged)"
    elif ratio >= 14.0:
        stage = "🟠 고령사회 (Aged)"
    elif ratio >= 7.0:
        stage = "🟡 고령화사회 (Aging)"
    else:
        stage = "🟢 젊은 사회"
        
    st.subheader(f"📌 {selected_region} 실시간 핵심 지표 요약")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("👥 총 인구수", f"{reg_data['총인구']:,} 명")
    col2.metric("👵 고령인구 비중 (65세+)", f"{reg_data['고령화비율']:.2f}%", stage)
    col3.metric("🧒 청소년 비중 (0~18세)", f"{reg_data['청소년비율']:.2f}%")
    col4.metric("📈 노령화지수", f"{reg_data['노령화지수']:.1f}")
    col5.metric("🎯 중위 연령", f"만 {reg_data['중위연령']} 세")
    
    st.markdown("---")
    
    # 메인 레이아웃 탭 분할
    tab_map, tab_pyramid, tab_compare = st.tabs(["🗺️ 인터랙티브 행정지도", "📊 세부 연령 구조 (피라미드)", "📈 지역 간 비교 및 랭킹"])
    
    # =========================================================================
    # TAB 1: 지도 시각화 (SVG 동적 매핑 기술 적용)
    # =========================================================================
    with tab_map:
        st.subheader(f"🗺️ 지도로 보는 {selected_label} 공간 분포")
        
        map_col1, map_col2 = st.columns([7, 3])
        
        # 색상 변환 헬퍼 함수
        def get_rgb_color(val, min_v, max_v, theme):
            if max_v == min_v:
                f = 0.0
            else:
                f = (val - min_v) / (max_v - min_v)
            f = max(0.0, min(1.0, f))
            
            # 테마별 색상 보간 코드
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
                    # 데이터 전처리 (광역 자치 단체만 필터링)
                    df_sido = df_metrics[df_metrics['region_level'] == 'Sido']
                    min_val = df_sido[selected_metric].min()
                    max_val = df_sido[selected_metric].max()
                    
                    soup = BeautifulSoup(svg_national, "xml")
                    soup.svg['width'] = '100%'
                    soup.svg['height'] = '550px'
                    
                    # CSS 효과 주입
                    style = soup.new_tag("style")
                    style.string = "path, polyline { transition: fill 0.3s; cursor:pointer; } path:hover, polyline:hover { fill-opacity: 0.8 !important; stroke: #333 !important; stroke-width: 2.5px !important; }"
                    soup.svg.insert(0, style)
                    
                    # 영문 ID와 한글 시도명 매칭용 맵
                    id_mapping = {
                        'seoul': '서울', 'busan': '부산', 'daegu': '대구', 'incheon': '인천', 'gwangju': '광주',
                        'daejeon': '대전', 'ulsan': '울산', 'sejong': '세종', 'gyeonggi': '경기', 'gangwon': '강원',
                        'chungbuk': '충북', 'chungnam': '충남', 'jeonbuk': '전북', 'jeonnam': '전남', 'gyeongbuk': '경북',
                        'gyeongnam': '경남', 'jeju': '제주'
                    }
                    
                    for path in soup.find_all(['path', 'polyline']):
                        p_id = path.get('id')
                        if p_id:
                            clean_id = p_id.lower().replace('-do', '').replace('-si', '').replace('special', '').strip()
                            kor_prefix = id_mapping.get(clean_id, '')
                            
                            # 일치하는 데이터 찾아서 색칠하기
                            matched_row = df_sido[df_sido['clean_region'].str.startswith(kor_prefix)] if kor_prefix else None
                            if matched_row is not None and not matched_row.empty:
                                r_data = matched_row.iloc[0]
                                val = r_data[selected_metric]
                                color = get_rgb_color(val, min_val, max_val, selected_theme)
                                
                                path['fill'] = color
                                path['stroke'] = '#ffffff'
                                path['stroke-width'] = '1.2px'
                                
                                # 브라우저 툴팁 주입
                                tooltip = soup.new_tag("title")
                                tooltip.string = f"{r_data['clean_region']}\n- {selected_label}: {val:.2f if isinstance(val, float) else val:,}"
                                path.append(tooltip)
                                
                    import streamlit.components.v1 as components
                    components.html(str(soup), height=570)
                else:
                    st.error("국가 지도 SVG 로딩 실패")
                    
            # 서울 지도 렌더링
            with map_tab2:
                svg_seoul = fetch_svg("Seoul_districts.svg")
                if svg_seoul:
                    df_seoul = df_metrics[df_metrics['clean_region'].str.startswith("서울특별시 ")]
                    min_val_s = df_seoul[selected_metric].min()
                    max_val_s = df_seoul[selected_metric].max()
                    
                    soup_s = BeautifulSoup(svg_seoul, "xml")
                    soup_s.svg['width'] = '100%'
                    soup_s.svg['height'] = '550px'
                    
                    style_s = soup_s.new_tag("style")
                    style_s.string = "path { transition: fill 0.3s; cursor:pointer; } path:hover { fill-opacity: 0.8 !important; stroke: #111 !important; stroke-width: 3.5px !important; }"
                    soup_s.svg.insert(0, style_s)
                    
                    # 영문 ID와 서울시 자치구명 매치
                    seoul_id_mapping = {
                        'jongno': '종로구', 'jung': '중구', 'yongsan': '용산구', 'seongdong': '성동구', 'gwangjin': '광진구',
                        'dongdaemun': '동대문구', 'jungnang': '중랑구', 'seongbuk': '성북구', 'gangbuk': '강북구', 'dobong': '도봉구',
                        'nowon': '노원구', 'eunpyeong': '은평구', 'seodaemun': '서대문구', 'mapo': '마포구', 'yangcheon': '양천구',
                        'gangseo': '강서구', 'guro': '구로구', 'geumcheon': '금천구', 'yeongdeungpo': '영등포구', 'dongjak': '동작구',
                        'gwanak': '관악구', 'seocho': '서초구', 'gangnam': '강남구', 'songpa': '송파구', 'gangdong': '강동구'
                    }
                    
                    for path in soup_s.find_all('path'):
                        p_id = path.get('id')
                        if p_id:
                            clean_id = p_id.lower().replace('-gu', '').replace('_gu', '').strip()
                            kor_district = seoul_id_mapping.get(clean_id, '')
                            
                            matched_row = df_seoul[df_seoul['clean_region'].str.contains(kor_district)] if kor_district else None
                            if matched_row is not None and not matched_row.empty:
                                r_data = matched_row.iloc[0]
                                val = r_data[selected_metric]
                                color = get_rgb_color(val, min_val_s, max_val_s, selected_theme)
                                
                                path['fill'] = color
                                path['stroke'] = '#ffffff'
                                path['stroke-width'] = '1.5px'
                                
                                tooltip = soup_s.new_tag("title")
                                tooltip.string = f"{r_data['clean_region']}\n- {selected_label}: {val:.2f if isinstance(val, float) else val:,}"
                                path.append(tooltip)
                                
                    components.html(str(soup_s), height=570)
                else:
                    st.error("서울 지도 SVG 로딩 실패")
                    
        with map_col2:
            st.markdown(f"### 🎨 지도 범례 ({selected_label})")
            
            # 선택된 테마에 따른 범례 바 생성
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
        
        pyr_col1, pyr_col2 = st.columns(2)
        
        # 1. 인구 피라미드 그리기
        with pyr_col1:
            st.markdown("#### 👨‍👩‍👧‍👦 남녀 성별 인구 피라미드")
            
            raw_row = reg_data['raw_row']
            male_cols = reg_data['male_cols']
            female_cols = reg_data['female_cols']
            ages = reg_data['ages']
            
            m_vals = [raw_row[male_cols[a]] for a in ages]
            f_vals = [raw_row[female_cols[a]] for a in ages]
            
            fig_pyramid = go.Figure()
            # 남성 가로막대 그리기 (좌측으로 뻗어나가도록 음수 부호 처리)
            fig_pyramid.add_trace(go.Bar(
                y=ages, x=[-v for v in m_vals],
                name='남성 (Male)', orientation='h',
                marker=dict(color='#1177b4'),
                hoverinfo='text',
                hovertext=[f"남성 만 {a}세: {v:,}명" for a, v in zip(ages, m_vals)]
            ))
            # 여성 가로막대 그리기
            fig_pyramid.add_trace(go.Bar(
                y=ages, x=f_vals,
                name='여성 (Female)', orientation='h',
                marker=dict(color='#e377c2'),
                hoverinfo='text',
                hovertext=[f"여성 만 {a}세: {v:,}명" for a, v in zip(ages, f_vals)]
            ))
            
            # X축 레이블을 양수로 치환
            max_p = max(max(m_vals), max(f_vals))
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
            
        # 2. 3대 연령군 구성 원형차트
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
            
            # 구 이름만 축약 표시 ("서울특별시 종로구" -> "종로구")
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

else:
    # 데이터가 로드되지 않은 상태에서 보여주는 기본 가이드
    st.info("👈 왼쪽 사이드바에서 데이터 분석을 시작해 주세요! 로컬에 CSV 파일이 위치하거나, 브라우저에서 직접 CSV 파일을 Drag & Drop으로 업로드하면 인구구조 및 대화형 지도 대시보드가 자동으로 로드됩니다.")
    
    st.markdown("---")
    st.markdown("### 📋 업로드할 행정안전부 주민등록 인구통계 다운로드 가이드")
    st.markdown("""
    1. **[행정안전부 주민등록 인구통계 홈페이지](https://jumin.mois.go.kr/)**에 접속합니다.
    2. 상단 메뉴에서 **[연령별 인구현황]**을 선택합니다.
    3. 조회조건을 아래와 같이 맞춘 뒤 **[조회]**를 누르고, 우측 하단의 **[CSV 다운로드]**를 선택해 주세요.
       * **구분:** `구분(남/여)` 선택 필수
       * **연령구분:** `1세 단위` 지정
       * **연령범위:** `0` 세부터 `100` 세 이상까지 지정
    4. 다운로드된 파일을 왼쪽 파일 업로더에 드롭해 주시면 본 대시보드가 실시간 연동되어 분석을 가동합니다!
    """)
