'userID'                 : 유저ID  
'big_assessmentItemID'   : 대분류  
'testId'                 : 시험지ID  
'assessmentItemID'       : 문제ID  
'KnowledgeTag'           : 태그ID   
'answerCode'             : 유저별 정답 여부 (0/1 binary)  
'Timestamp'              : 문제를 풀기 시작한 시간  

'Timestamp_unix'         : Timestamp -> 유닉스 시간  
'lag_time'               : 문제와 문제 사이의 시간  

'acc_avg_item'           : 문제별 평균 정답률  
'relative_answerCode'    : 유저별 상대적 정답률 (유저별 정답 여부(answerCode) - 문제별 평균 정답률(acc_avg_item))  
----'mix_u_big'              : 유저별 대분류 정답률 피처를 생성하기 위한 피처(훈련에 사용 x)  
----'mix_u_test'             : 유저별 시험지 정답률 피처를 생성하기 위한 피처(훈련에 사용 x)  
----'mix_u_item'             : 유저별 문제 정답률 피처를 생성하기 위한 피처(훈련에 사용 x)  
----'mix_u_tag'              : 유저별 태그 정답률 피처를 생성하기 위한 피처(훈련에 사용 x)  
'acc_by_user_big'        : 유저별 대분류 정답률  
'acc_by_user_test'       : 유저별 시험지 정답률  
----'acc_by_user_item'       : 유저별 문제 정답률 (확인해보니 풀었던 문제를 또 푸는 유저는 없었기 때문에 해당 피처는 answerCode와 동일함. 훈련에 사용 X)  
'acc_by_user_tag'        : 유저별 태그 정답률  
'p_item_answer_cnt'      : 유저별 이전 문제 정답 횟수 (LGBM_baseline의 user_correct_answer 피처와 동일)  
'p_item_cnt'             : 유저별 이전에 푼 문제 수   (LGBM_baseline의 user_total_answer 피처와 동일)  
'p_item_acc'             : 유저별 이전 문제 정답률    (LGBM_baseline의 user_acc 피처와 동일)  
'p_item_relative_acc_sum': 유저별 이전 문제의 상대적인 정답률_sum  
'p_item_relative_acc'    : 유저별 이전 문제의 상대적인 정답률  
'p_big_frequency_cnt'    : 유저별 해당 대분류를 이전에 몇번 풀었는지  
'p_test_frequency_cnt'   : 유저별 해당 시험지를 이전에 몇번 풀었는지  
'p_item_frequency_cnt'   : 유저별 해당 문제를 이전에 몇번 풀었는지  
'p_tag_frequency_cnt'    : 유저별 해당 태그를 이전에 몇번 풀었는지  

'entire_acc_by_big'      : 해당 대분류의 정답률(전체 기준)  
'entire_user_cnt_by_big' : 해당 대분류를 푼 유저 수(전체 기준)  
'entire_acc_by_test'     : 해당 시험지의 정답률(전체 기준)       
'entire_user_cnt_by_test': 해당 시험지를 푼 유저 수(전체 기준)   
'entire_acc_by_item'     : 해당 문제의 정답률(전체 기준)       (LGBM_baseline의 test_mean 피처와 동일)  
'entire_user_cnt_by_item': 해당 문제를 푼 유저 수(전체 기준)   (LGBM_baseline의 test_sum 피처와 동일)  
'entire_acc_by_tag'      : 해당 태그의 정답률(전체 기준)       (LGBM_baseline의 tag_mean 피처와 동일)  
'entire_user_cnt_by_tag' : 해당 태그를 푼 유저 수(전체 기준)   (LGBM_baseline의 user_ctag_sumorrect_answer 피처와 동일)  