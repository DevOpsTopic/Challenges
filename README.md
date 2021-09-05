# Topics of DevOps Challenges in Developer Discussions


## DevOps Development
DevOps practices combine software development and IT (Information Technology) operations. Our continuous needs for rapid but quality software development necessitate the adoption of high quality
DevOps tools. Therefore, it is important to learn the challenges DevOps developers face while using the currently available tools and techniques. The online developer forum Stack Overflow (SO) is popular among DevOps developers. We observed a growing number of posts in SO regarding discussions. However, we are aware of no previous study that analyzed SO posts to learn the challenges DevOps developers discuss in SO.


## Benchmark Dataset
We download Stack Overflow June 2020 data dump for this study. Then we extracted DevOps related posts based on our tag list. Our tag list contain 19 tags ("ansible","azure-devops","devops","jenkins","kubernetes","terraform","chef","continuous-integration","ibm-cloud","gitlab","jenkins-pipeline","gitlab-ci","puppet","amazon-cloudformation","pipeline","azure-pipelines","jenkins-plugins","continuous-deployment","devops-services"). Our dataset includes 49,139,907 questions and answers posted over the last 13
years from August 2008 to June 2020. 

## Folder With Details
In this replication package, we include all the data, codes that we used in this study. <br/>
Folder “Develop TagSet” contains
1.	Tag selection and tag set construction queries -/Develop/TagSet/queries.txt
2.	Candidate tags, significance & relevance with tags - -/Develop/TagSet/Develop_Devops_dataset_Tag.xlsx
3.	DevOps tag sets with significance & relevance - -/Develop/TagSet/Devops_Tag_sets.txt
4.	DevOps post extract query - -/Develop/TagSet/Dataset_Built_Query.txt
Folder Dataset contain Stack overflow and DevOps data download link.
Folder “Topic_modeling_Code” contain develop topic, topic popularity, difficulty (devops_topic_modeling.py) and coherence score calculation (Topics_Ranking_Graph.py) code.
Folder “DevOpsTopic” contain DevOps topic, category, sub-category (evOps Topic Category and Sub-Category.xlsx) and Coherence score under this folder (DevOpsTopic/CoherenceScore)

 
