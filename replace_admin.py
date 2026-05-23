from pathlib import Path

path = Path('main_app.py')
text = path.read_text(encoding='utf-8')
start = text.find('def admin_page(db):')
end = text.find('def classifier_page(db):', start)
if start == -1 or end == -1:
    raise SystemExit('markers not found')
new = '''def admin_page(db):
    users_df = db.get_all_users()
    if not users_df.empty:
        st.dataframe(users_df)
        selected_user = st.selectbox("", users_df['username'].tolist())
        if selected_user and selected_user.lower() != 'admin':
            selected_row = users_df[users_df['username'] == selected_user].iloc[0]
            selected_user_id = int(selected_row['id'])
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Delete", key="delete_user"):
                    db.delete_user(selected_user_id)
                    st.rerun()
            with col2:
                if st.button("Block", key="block_user"):
                    db.block_user(selected_user_id)
                    st.rerun()
    classifications_df = db.get_all_classifications()
    if not classifications_df.empty:
        st.dataframe(classifications_df)
    system_stats = db.get_system_stats()
    spam_count = 0
    not_spam_count = 0
    if not system_stats['spam_ham_counts'].empty:
        for _, row in system_stats['spam_ham_counts'].iterrows():
            if row['prediction'] == 'Spam':
                spam_count = row['count']
            elif row['prediction'] == 'Not Spam':
                not_spam_count = row['count']
    st.dataframe(pd.DataFrame([
        ['total_messages', system_stats['total_classifications']],
        ['spam', spam_count],
        ['not_spam', not_spam_count]
    ], columns=['metric', 'value']))
    feedback_df = db.get_feedback()
    if not feedback_df.empty:
        st.dataframe(feedback_df)


def classifier_page(db):'''
path.write_text(text[:start] + new + text[end:], encoding='utf-8')
print('replaced', start, end)
