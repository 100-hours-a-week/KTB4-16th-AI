-- 백엔드용 읽기 전용 계정 (기능6 뷰 2개만 조회 가능)
-- 클라우드 팀이 AI Postgres 관리자 계정으로 1회 실행. 비밀번호는 실행할 때 정하고
-- GitHub Secrets 등으로 백엔드에 전달 (이 파일·채팅·저장소에 적지 않기).
-- 마이그레이션 0008 적용 후 실행해야 뷰가 있다.

CREATE ROLE mulo_be_reader LOGIN PASSWORD :'be_reader_password';

-- 원본 테이블에는 권한을 주지 않고 뷰 두 개만 SELECT
GRANT CONNECT ON DATABASE muro_ai TO mulo_be_reader;
GRANT USAGE ON SCHEMA public TO mulo_be_reader;
GRANT SELECT ON be_user_music_genres, be_vote_questions TO mulo_be_reader;

-- 실행 예: psql "$ADMIN_URL" -v be_reader_password="$BE_READER_PASSWORD" -f be_readonly_role.sql
-- DB 이름이 muro_ai가 아니면 GRANT CONNECT 줄의 이름을 바꿔서 실행
