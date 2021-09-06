# Topics of DevOps Challenges in Developer Discussions


## DevOps Development
DevOps practices combine software development and IT (Information Technology) operations. Our continuous needs for rapid but quality software development necessitate the adoption of high quality
DevOps tools. Therefore, it is important to learn the challenges DevOps developers face while using the currently available tools and techniques. The online developer forum Stack Overflow (SO) is popular among DevOps developers. We observed a growing number of posts in SO regarding discussions. However, we are aware of no previous study that analyzed SO posts to learn the challenges DevOps developers discuss in SO.


## Benchmark Dataset
We download Stack Overflow June 2020 data dump for this study. Then we extracted DevOps related posts using our tag list. The final list of 19 devops tags that we used to collect our devops posts from Stack Overflow. Devops tags are ("ansible","azure-devops","devops","jenkins","kubernetes","terraform","chef","continuous-integration","ibm-cloud","gitlab","jenkins-pipeline","gitlab-ci","puppet","amazon-cloudformation","pipeline","azure-pipelines","jenkins-plugins","continuous-deployment","devops-services"). 

## Folder With Details
In this replication package, we include all the data, codes that we used in this study. <br/>
1. Folder “Develop TagSet” contains
i.	Tag selection and tag set construction queries -/Develop/TagSet/queries.txt
ii.	Candidate tags, significance & relevance with tags - -/Develop/TagSet/Develop_Devops_dataset_Tag.xlsx
iii.	DevOps tag sets with significance & relevance - -/Develop/TagSet/Devops_Tag_sets.txt
iv.	DevOps post extract query - -/Develop/TagSet/Dataset_Built_Query.txt  
v. Category and topic evolution query - TopicsEvolve.sql & categoriesevolve.sql<br/>
2. Folder Dataset contain Stack overflow and DevOps data download link.
3. Folder “Topic_modeling_Code” contain develop topic, topic popularity, difficulty (devops_topic_modeling.py) and coherence score calculation (Topics_Ranking_Graph.py) code.  <br/>
4. Folder “DevOpsTopic” contain DevOps topic, category, sub-category (evOps Topic Category and Sub-Category.xlsx), topic difficulty popularity, bubble chart, topic evoluation and Coherence score under this folder (DevOpsTopic/CoherenceScore)  <br/>
5. Folder "DevOpsSurvey" contain survey link and devops survey data 
6. Folder "DevOpsPhases" contain manually levelling devops phases  

 
