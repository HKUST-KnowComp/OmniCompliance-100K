
for file in ../data/rules/policys/github/json/*.json; do
    python agent_case_research_general.py \
    --rule_file_path $file \
    --output_file_path ../data/cases/policys/github \
    --rule_catagory '(github - policy)' 
done

for file in ../data/rules/privacy/hipaa/json/*.json; do
    python agent_case_research_general.py \
    --rule_file_path $file \
    --output_file_path ../data/cases/privacy/hipaa \
    --rule_catagory '(regulation - hipaa)' 
done

for file in ../data/rules/privacy/gdpr/json/*.json; do
    python agent_case_research_general.py \
    --rule_file_path $file \
    --output_file_path ../data/cases/privacy/gdpr \
    --rule_catagory '(regulation - gdpr)' 
done

for file in ../data/rules/privacy/ccpa/json/*.json; do
    python agent_case_research_general.py \
    --rule_file_path $file \
    --output_file_path ../data/cases/privacy/ccpa \
    --rule_catagory '(regulation - ccpa)' 
done

for file in ../data/rules/privacy/chinese_law/json/*.json; do
    python agent_case_research_general.py \
    --rule_file_path $file \
    --output_file_path ../data/cases/privacy/chinese_law \
    --rule_catagory '(regulation - chinese_law)' 
done


for file in ../data/rules/privacy/data_act/json/*.json; do
    python agent_case_research_general.py \
    --rule_file_path $file \
    --output_file_path ../data/cases/privacy/data_act \
    --rule_catagory '(regulation - data_act)' 
done

for file in ../data/rules/privacy/eu_ai_act/json/*.json; do
    python agent_case_research_general.py \
    --rule_file_path $file \
    --output_file_path ../data/cases/privacy/eu_ai_act \
    --rule_catagory '(regulation - eu_ai_act)' 
done

for file in ../data/rules/privacy/sb35/json/*.json; do
    python agent_case_research_general.py \
    --rule_file_path $file \
    --output_file_path ../data/cases/privacy/sb35 \
    --rule_catagory '(regulation - sb35)' 
done

for file in ../data/rules/finance/json/*.json; do
    python agent_case_research_general.py \
    --rule_file_path $file \
    --output_file_path ../data/cases/finance \
    --rule_catagory '(eu regulation - finance)' 
done

for file in ../data/rules/medical/json/*.json; do
    python agent_case_research_general.py \
    --rule_file_path $file \
    --output_file_path ../data/cases/medical \
    --rule_catagory '(eu regulation - medical)' 
done

for file in ../data/rules/cyber_security/mitre_attack/json/*.json; do
    python agent_case_research_general.py \
    --rule_file_path $file \
    --output_file_path ../data/cases/cyber_security/mitre_attack \
    --rule_catagory '(cyber security)' 
done

for file in ../data/rules/edu/academic_integrity/json/*.json; do
    python agent_case_research_general.py \
    --rule_file_path $file \
    --output_file_path ../data/cases/edu/academic_integrity \
    --rule_catagory '(education - academic integrity)' 
done

for file in ../data/rules/edu/discrimination_us_edu_dept/json/*.json; do
    python agent_case_research_general.py \
    --rule_file_path $file \
    --output_file_path ../data/cases/edu/discrimination_us_edu_dept \
    --rule_catagory '(education - bias and discrimination)' 
done


for file in ../data/rules/edu/online_learning/json/*.json; do
    python agent_case_research_general.py \
    --rule_file_path $file \
    --output_file_path ../data/cases/edu/online_learning \
    --rule_catagory '(education - online learning)' 
done

for file in ../data/rules/foundation_rights/json/*.json; do
    python agent_case_research_general.py \
    --rule_file_path $file \
    --output_file_path ../data/cases/foundation_rights \
    --rule_catagory '(eu - foundation rights)' 
done

for file in ../data/rules/policys/google/json/*.json; do
    python agent_case_research_general.py \
    --rule_file_path $file \
    --output_file_path ../data/cases/policys/google \
    --rule_catagory '(google - policy)' 
done

for file in ../data/rules/policys/openai/json/*.json; do
    python agent_case_research_general.py \
    --rule_file_path $file \
    --output_file_path ../data/cases/policys/openai \
    --rule_catagory '(openai - policy)' 
done

for file in ../data/rules/policys/reddit/json/*.json; do
    python agent_case_research_general.py \
    --rule_file_path $file \
    --output_file_path ../data/cases/policys/reddit \
    --rule_catagory '(reddit - policy)' 
done

for file in ../data/rules/policys/wechat/json/*.json; do
    python agent_case_research_general.py \
    --rule_file_path $file \
    --output_file_path ../data/cases/policys/wechat \
    --rule_catagory '(wechat - policy)' 
done

for file in ../data/rules/policys/x/json/*.json; do
    python agent_case_research_general.py \
    --rule_file_path $file \
    --output_file_path ../data/cases/policys/x \
    --rule_catagory '(x - policy)' 
done




