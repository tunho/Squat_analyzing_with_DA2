import streamlit as st
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objects as go
import os
import time

# Import our customized robust logic core
from squat_analyzer_knee import SquatAnalyzer

st.set_page_config(layout="wide", page_title="3D Squat Web Viewer")

st.title("Rigid Body Physics: Interactive 3D Squat Web Viewer")
st.markdown("왼쪽은 MediaPipe가 관측한 2D 영상이며, 오른쪽은 허벅지/정강이 **100% 길이 보존 강체 역학(Rigid Body Kinematics)**과 **Depth Anything V2**가 결합되어 오차가 완벽히 통제된 3D 뼈대입니다.")

# 1. 사이드바 컨트롤 & 비디오 로드 
st.sidebar.header("Controls & Video")
video_files = [f for f in os.listdir("dataset/true") if f.endswith(".mp4")]
# 기본 재생 영상을 true_5.mp4로 지정
try:
    default_idx = video_files.index("true_5.mp4")
except ValueError:
    default_idx = 0

selected_video = st.sidebar.selectbox("Select Video", video_files, index=default_idx)
video_path = os.path.join("dataset/true", selected_video)

# 2. 프로세싱 (캐싱하여 영상 당 한 번만 연산)
@st.cache_resource
def process_video_and_get_data(path):
    analyzer = SquatAnalyzer()
    analyzer.process_video(path, show_window=False)
    _, _, params = analyzer.finalize_analysis()
    return analyzer, params

if not os.path.exists(video_path):
    st.error(f"Video not found: {video_path}")
    st.stop()

with st.spinner("AI 딥러닝 3차원 공간 해석 중... (Depth Anything V2)"):
    analyzer, params = process_video_and_get_data(video_path)

st.sidebar.success("해석 완료!")

# ==========================================
# 3. 데이터 추출 및 설정 
# ==========================================
history = analyzer.history_data
num_frames = len(history)

# 애니메이션 상태 관리
if 'playing' not in st.session_state:
    st.session_state.playing = False
if 'frame_idx' not in st.session_state:
    st.session_state.frame_idx = 0

def toggle_play():
    st.session_state.playing = not st.session_state.playing

st.sidebar.button("▶️ Play / ⏸️ Pause", on_click=toggle_play)

# Streamlit 위젯은 key로 묶어주어야 상태가 자동으로 갱신됩니다.
st.sidebar.slider("Timeline (프레임 스크롤)", 0, num_frames - 1, key='frame_idx')

base_k = params.get('base_k', 1.0)
obs_max_thigh = max(f['thigh_len_2d'] for f in history)
obs_max_shank = max(f['shank_len_2d'] for f in history)
final_thigh_len = max(obs_max_thigh, obs_max_shank * 1.2)
final_shank_len = final_thigh_len / 1.2
stable_radii = analyzer.corrector.get_radii(final_thigh_len, final_shank_len)

frame = history[st.session_state.frame_idx]

hip_offset = st.sidebar.slider("Hip Z (Depth) 수동 조정 (시각 디버깅용)", -50.0, 50.0, 0.0)

# ==========================================
# 4. 화면 분할 (좌: 영상 / 우: Plotly 3D)
# ==========================================
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("2D Tracking (MediaPipe)")
    if frame.get('drawn_image') is not None:
        rgb_frame = frame['drawn_image'][..., ::-1] # BGR to RGB
        st.image(rgb_frame, use_container_width=True)

with col2:
    st.subheader("Interactive 3D Simulation")
    
    # --- 핵심: 메인 엔진과 100% 동일한 피타고라스 역산 공식 ---
    # [A] 무릎 앵커 
    knee_z_metric = frame['knee']['z'] * base_k
    knee_z_center = knee_z_metric + stable_radii['knee']
    p_final_knee = np.array([frame['knee']['x'], frame['knee']['y'], knee_z_center])

    # [B] 발목 역산 (무릎에서 뻣어나감)
    shank_len = np.linalg.norm([frame['ankle']['x'] - frame['knee']['x'], frame['ankle']['y'] - frame['knee']['y']])
    if final_shank_len > shank_len:
         z_needed_shank = np.sqrt(final_shank_len**2 - shank_len**2)
    else:
         z_needed_shank = 0.0
         
    direction_shank = 1 if frame['ankle']['z'] > frame['knee']['z'] else -1
    ankle_z_center = knee_z_center + (direction_shank * z_needed_shank)
    p_final_ankle = np.array([frame['ankle']['x'], frame['ankle']['y'], ankle_z_center])
    
    # [C] 골반 역산 (무릎에서 뻣어나감)
    thigh_len = np.linalg.norm([frame['hip']['x'] - frame['knee']['x'], frame['hip']['y'] - frame['knee']['y']])
    if final_thigh_len > thigh_len:
         z_needed_thigh = np.sqrt(final_thigh_len**2 - thigh_len**2)
    else:
         z_needed_thigh = 0.0
         
    direction_hip = 1 if frame['hip']['z'] > frame['knee']['z'] else -1
    hip_z_center = knee_z_center + (direction_hip * z_needed_thigh)
    p_final_hip = np.array([frame['hip']['x'], frame['hip']['y'], hip_z_center + hip_offset])

    # 시각화 영점 조정 (발목을 기준점 0,0,0으로 강제 이동) 
    origin = p_final_ankle.copy()
    def to_plot(pt):
        # MediaPipe: X(우), Y(하), Z(뒤) -> Plotly: X(우), Y(앞뒤 깊이, Plotly에선 Y가 화면 평면), Z(위)
        return [pt[0] - origin[0], pt[2] - origin[2], -(pt[1] - origin[1])]

    vis_hip = to_plot(p_final_hip)
    vis_knee = to_plot(p_final_knee)
    vis_ankle = to_plot(p_final_ankle)

    x_pts = [vis_hip[0], vis_knee[0], vis_ankle[0]]
    # Plotly에서 y는 평면의 앞뒤(Depth), z는 상하(Height)를 담당합니다.
    y_pts = [vis_hip[1], vis_knee[1], vis_ankle[1]] 
    z_pts = [vis_hip[2], vis_knee[2], vis_ankle[2]]

    # 3D 궤적 생성
    fig = go.Figure()
    
    # 뼈 막대기
    fig.add_trace(go.Scatter3d(
        x=x_pts, y=y_pts, z=z_pts,
        mode='lines+markers',
        marker=dict(
            size=[10, 8, 8], 
            color=['red', 'blue', 'orange'], 
            symbol='circle'
        ),
        line=dict(color='yellow', width=8),
        name='Skeleton'
    ))

    # 레이아웃(카메라 방향) 고정 설정 
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-300, 300], title="좌/우 (X)"),
            yaxis=dict(range=[-200, 400], title="앞/뒤 깊이 (Depth Z)"),
            zaxis=dict(range=[0, 600], title="높이 (Y)"),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=0.5) # 대각선 측면에서 비스듬히 보는 디폴트 앵글
            )
        ),
        uirevision='constant', # 프레임이 업데이트되어도 사용자가 돌려놓은 3D 앵글이 초기화되지 않음!
        margin=dict(r=10, l=10, b=10, t=10),
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# 5. 디버깅 지표 표시
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 실시간 역학 지표")
calc_thigh_len = np.linalg.norm(p_final_hip - p_final_knee)
calc_shank_len = np.linalg.norm(p_final_knee - p_final_ankle)
raw_hip_z = frame['hip']['z'] * base_k

st.sidebar.write(f"- **순수 환율 (K):** {base_k:.3f}")
st.sidebar.write(f"- **강체 허벅지 길이:** {calc_thigh_len:.1f} (Target: {final_thigh_len:.1f})")
st.sidebar.write(f"- **강체 정강이 길이:** {calc_shank_len:.1f} (Target: {final_shank_len:.1f})")
st.sidebar.write(f"- **골반 뼈 깊이 변화량 (Rigid - Raw):** {(p_final_hip[2] - hip_offset) - raw_hip_z:.1f} 픽셀")

# 6. 애니메이션 재생 루프 (자동 반복)
if st.session_state.playing:
    time.sleep(0.04) # 1 프레임 당 약 0.04초 대기 (25 fps 내외)
    st.session_state.frame_idx = (st.session_state.frame_idx + 1) % num_frames
    st.rerun()
