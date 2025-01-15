import streamlit as st

st.set_page_config(
    page_title="Ex-stream-ly Cool App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)


def get_user_name():
    return 'John'


with st.echo():
    # Everything inside this block will be both printed to the screen
    # and executed.

    def get_punctuation():
        return '!!!'

    greeting = "Hi there, "
    value = get_user_name()
    punctuation = get_punctuation()

    st.write(greeting, value, punctuation)

# And now we're back to _not_ printing to the screen
foo = 'bar'
st.write('Done!')


# import streamlit as st

# # 두 개의 컬럼 생성 (메인 영역과 사이드바 영역)
# main_col, sidebar_col = st.columns([0.7, 0.3])  # 비율을 조절하여 크기 조정 가능

# # 메인 컬럼에 내용 추가
# with main_col:
#     st.title("메인 컨텐츠")
#     st.write("여기에 주요 내용을 작성하세요")

# # 오른쪽 사이드바 컬럼에 내용 추가
# with sidebar_col:
#     st.title("사이드바")
#     st.selectbox("옵션 선택", ["옵션1", "옵션2", "옵션3"])
#     st.button("실행")
