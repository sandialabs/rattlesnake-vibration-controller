---
numbering:
 headings: false
---
(sec:sqa)=
# Software Quality Assurance

Rattlesnake began its life as a research code with very few resources, so initial development priorities were around capability development. The initial goal of the project was to simply be a sandbox to allow researchers to try out ideas. However, due to its initial successes and the growing comfort of its user base, it has started to be used for more important testing activities as a full-fledged vibration control software package, including driving expensive shaker systems with high-value test articles. Due to the consequences of Rattlesnake misbehaving during one of these expensive tests, the developers of Rattlesnake have begun to put more effort into Software Quality Assurance.

To provide a quality assurance framework for the Rattlesnake Vibration Controller, the development team has targeted the Advanced Simulation and Computing (ASC) Software Quality Plan [(Turgeon, 2019)](https://doi.org/10.2172/1762331). The plan defines 12 process areas and 30 software quality engineering best practices. The ASC Software Quality Plan adopts a graded approach to software quality using "Levels of Formality" (LOF), which is a measure of how "formal" the evaluation of the software needs to be based on the risks and consequences of bad software behavior. For example, a small research code to implement an algorithm from a paper need not be evaluated to the same level of rigor as a software product that operates a nuclear power plant. The level of formality defines which practices need to be followed and to what level. This framework is useful for the resource-limited developers of Rattlesnake; it helps highlight the most important issues to tackle first and gives useful metrics for evaluating progress.

All assessments made in this document are done by the Rattlesnake Development Team.

## ASC SQE Practices & Process Areas  

The software plan is divided into four categories: @sec:sqa_project_management, @sec:sqa_software_engineering, @sec:sqa_software_verification, and @sec:sqa_training.  Each category is divided into sub-categories, which contain software engineering practices.  In total, there are 30 software engineering practices that the ASC Software Quality Plan identifies.  Depending on the target LOF, the ASC Software Quality Plan identifies the score that the software should strive to reach.  Scores are defined in @sec:sqa_scores.

(sec:sqa_project_management)=
## Project Management

Project Management consists of balancing the work to be done with resources available.

### Integrated Teaming

Integrated teaming defines the project, its mission, the project members and their responsibilities, the users and stakeholders, and interfaces with other projects.

#### PR1. Document and maintain a strategic plan.

The software does not currently have a strategic plan.

|  Low LOF  |  Med LOF  |  High LOF  | Current Status Grade  | Current Priority  |
|----|----|----|----|------|
|  0 | 3  | 4  | 0  | Low  |


### Graded Level of Formality

Each project following the ASC Software Quality Plan assesses itself to define a Level of Formality in implementing the software practices.  This assessment considers factors like how the product should be used, the complexity of the project, and the consequences of issues with the software.

#### PR2. Perform a risk-based assessment, determine level of formality and applicable practices, and obtain approvals.

The risk-based assessment has not yet been done.

|  Low LOF  |  Med LOF  |  High LOF  | Current Status Grade  | Current Priority  |
|----|----|----|----|-------|
| 3  | 3  | 4  | 0  | High  |

### Measurement and Analysis

Process improvement is a continual activity throughout a product's lifecycle.  In order to improve processes, measurement, analysis, and evaluation of processes metrics is required.

#### PR3. Document, monitor, and control lifecycle processes and their interdependencies and obtain approvals.

The software has the beginnings of this through the use of GitHub issues, but it is not a formal process.

|  Low LOF  |  Med LOF  |  High LOF  | Current Status Grade  | Current Priority  |
|----|----|----|----|------|
|  0 | 3  | 4  | 1  | Low  |

#### PR4. Define, collect, and monitor appropriate process metrics.

The software has the beginnings of this through the use of GitHub issues, but it is not a formal process.

|  Low LOF  |  Med LOF  |  High LOF  | Current Status Grade  | Current Priority  |
|----|----|----|----|------|
|  0 | 3  | 4  | 1  | Low  |

#### PR5. Periodically evaluate quality issues and implement process improvements.  

There has been no evaluation of processes, development has been ad hoc as new features are needed.

|  Low LOF  |  Med LOF  |  High LOF  | Current Status Grade  | Current Priority  |
|----|----|----|----|----|
| 0  | 3  | 4  | 0  | Low  |

### Requirements Development and Management

Defining product requirements is a key part to making sure product development is pointed in the correct direction.

#### PR6. Identify stakeholders and other requirements sources.  

Stakeholders are identified through working groups, but not documented in any formal way.

|  Low LOF  |  Med LOF  |  High LOF  | Current Status Grade  | Current Priority  |
|----|----|----|----|----|
| 3  | 3  | 4  | 2  | Med  |

#### PR7. Gather and manage stakeholders' expectations, requirements, and constraints.  

Stakeholder requirements are communicated through ad hoc channels like email and teams messages.  Requests may be documented more formally through GitHub issues with feature request labels.

|  Low LOF  |  Med LOF  |  High LOF  | Current Status Grade  | Current Priority  |
|----|----|----|----|----|
| 3  | 3  | 4  | 1  | Med  |

#### PR8. Derive, negotiate, manage, and trace requirements.  

Requirements are traced in an ad hoc way back to test needs and communicated informally: "We need X feature for a test on Y date."

|  Low LOF  |  Med LOF  |  High LOF  | Current Status Grade  | Current Priority  |
|----|----|----|----|----|
| 3  | 3  | 4  | 1  | Low  | 

### Risk Management

Identifying and mitigating risks is an important aspect of project management.  Risks can turn into threats to the success of the project if not addressed proactively.

#### PR9. Identify and analyze risk events.  
 
No formal risk assessment has been done.  

|  Low LOF  |  Med LOF  |  High LOF  | Current Status Grade  | Current Priority  |
|----|----|----|----|----|
| 3  | 3  | 4  | 0  | Med  |

#### PR10. Define, monitor, and implement the risk response.  

No formal risk assessment has been done.  

|  Low LOF  |  Med LOF  |  High LOF  | Current Status Grade  | Current Priority  |
|----|----|----|----|----|
| 3  | 3  | 4  | 0  | Med  |

### Project Planning and Oversight

Project planning guides the implementation of the project while balancing project quality, cost, schedule, and performance.

#### PR11. Create and manage the project plan.

The software does not currently have any kind of project plan.

|  Low LOF  |  Med LOF  |  High LOF  | Current Status Grade  | Current Priority  |
|----|----|----|----|----|
| 3  | 3  | 4  | 0  | Med  |

#### PR12. Track project performance versus project plan and implement needed actions.

The software does not have a project plan.

|  Low LOF  |  Med LOF  |  High LOF  | Current Status Grade  | Current Priority  |
|----|----|----|----|----|
| 3  | 3  | 4  | 0  | Med  |

(sec:sqa_software_engineering)=
## Software Engineering

Software engineering is the specification, design, development, testing, operation, support, and eventual retirement of software.

### Technical Solution

The technical solution is the generation of a correctly working software product for the end users.  This includes design, implementation, documentation, as well as managing third-part dependencies.

#### PR13. Communicate and review design.

Design reviews are informal, often communicated through email.  These are often little more than an "it's working" message with a screenshot and little additional explanation.

|  Low LOF  |  Med LOF  |  High LOF  | Current Status Grade  | Current Priority  |
|----|----|----|----|----|
|  0 | 3  | 4  | 1  | Low  |

#### PR14. Create required software and product documentation.

Software has a fairly comprehensive user's manual, but no theory or design manual.  Documentation is in the form of a PDF which has linking and table of contents, and is reasonably searchable.

|  Low LOF  |  Med LOF  |  High LOF  | Current Status Grade  | Current Priority  |
|----|----|----|----|----|
|  0 | 3  | 4  | 3  | Med  |

#### PR15. Identify and track third party software products and follow applicable agreements.  

Third-party license agreements were identified during the open-source procedure but not documented well.

|  Low LOF  |  Med LOF  |  High LOF  | Current Status Grade  | Current Priority  |
|----|----|----|----|----|
|  0 | 3  | 4  | 2  | Low  |

#### PR16. Identify, accept ownership, and manage assimilation of other software products.  

We do not currently provide any support towards the third-party software packages that are used. 

|  Low LOF  |  Med LOF  |  High LOF  | Current Status Grade  | Current Priority  |
|----|----|----|----|----|
|  0 | 3  | 4  | 0  | Low  |

### Configuration Management

Configuration management provides a controlled environment for development, production, and support activities.

#### PR17. Perform version control of identified software product artifacts.  

Version control is currently done on the SandiaLabs GitHub page.  Software is tracked but artifacts are not. 

|  Low LOF  |  Med LOF  |  High LOF  | Current Status Grade  | Current Priority  |
|----|----|----|----|----|
| 4  | 4  | 4  | 3  | High  |

#### PR18. Record and track issues associated with the software product.  

Issues are tracked on the GitHub Page as well as an internal GitLab page.  There are many stale issues, unconfirmed bugs that are perhaps no longer applicable, and feature requests that have gone nowhere on the internal page.

|  Low LOF  |  Med LOF  |  High LOF  | Current Status Grade  | Current Priority  |
|----|----|----|----|----|
|  0 | 3  | 4  | 3  | High  |

#### PR19. Ensure backup and disaster recovery of software product artifacts.  

Software is available on the GitHub page.  Software is also backed up internally at Sandia National Laboratories.

|  Low LOF  |  Med LOF  |  High LOF  | Current Status Grade  | Current Priority  |
|----|----|----|----|----|
| 4  | 4  | 4  | 3  | High  |

### Integrated Product

The purpose of product integration is to manage the deployment of the product to the user in the form of a release.  The release should contain all required assets and artifacts.

#### PR20. Plan and generate the release package.

We push the software to GitHub, and it is up to the users to download it and use it.  This could be improved to better allow system maintainers to know when they should update their software.  Providing executables, etc. would be useful.  

|  Low LOF  |  Med LOF  |  High LOF  | Current Status Grade  | Current Priority  |
|----|----|----|----|----|
|  0 | 3  | 4  | 1  | High  |

#### PR21. Certify that the software product (code and its related artifacts) is ready for release and distribution.  

There is currently no testing or certification in the release process.

|  Low LOF  |  Med LOF  |  High LOF  | Current Status Grade  | Current Priority  |
|----|----|----|----|----|
|  0 | 3  | 4  | 0  | High  |

### Deployment and Lifecycle Support

Simply providing the software to users is typically not sufficient.  Training for installation and operation should also be provided.

#### PR22. Distribute release to customers.  

Users can download the software from GitHub, and we do use the "releases" capability periodically, but not in a very disciplined way. 

|  Low LOF  |  Med LOF  |  High LOF  | Current Status Grade  | Current Priority  |
|----|----|----|----|----|
|  0 | 3  | 4  | 1  | Low  |

#### PR23. Define and implement a customer support plan.  

Customer support is provided via GitHub issues, or otherwise informal email/teams communication.  There is no formal plan.  

|  Low LOF  |  Med LOF  |  High LOF  | Current Status Grade  | Current Priority  |
|----|----|----|----|----|
|  0 | 3  | 4  | 1  | Med  |

#### PR24. Implement the training identified in the customer support plan.  

There is ad hoc documentation, but no formal training for Rattlesnake

|  Low LOF  |  Med LOF  |  High LOF  | Current Status Grade  | Current Priority  |
|----|----|----|----|----|
|  0 | 3  | 4  | 1  | Low  |
#### PR25. Evaluate customer feedback to determine customer satisfaction.  

Feedback is currently ad hoc, word of mouth, emails, teams.  

|  Low LOF  |  Med LOF  |  High LOF  | Current Status Grade  | Current Priority  |
|----|----|----|----|----|
|  0 | 3  | 4  | 1  | Low  |

(sec:sqa_software_verification)=
## Software Verification

Verification and validation is necessary to ensure that the product released to the users behaves as specified in the requirements.

### Software Verification

#### PR26. Develop and maintain a software verification plan.  

Currently there is no verification plan  

|  Low LOF  |  Med LOF  |  High LOF  | Current Status Grade  | Current Priority  |
|----|----|----|----|----|
|  0 | 3  | 4  | 0  | High  |

#### PR27. Conduct tests to demonstrate that acceptance criteria are met and to ensure that previously tested capabilities continue to perform as expected.  

Currently testing is ad hoc; developer proves to themselves that the thing works then pushes it.

|  Low LOF  |  Med LOF  |  High LOF  | Current Status Grade  | Current Priority  |
|----|----|----|----|----|
|  0 | 3  | 4  | 1  | High  |

#### PR28. Conduct independent technical reviews to evaluate adequacy with respect to requirements.  

No independent reviews have been performed other than just people using the software.

|  Low LOF  |  Med LOF  |  High LOF  | Current Status Grade  | Current Priority  |
|----|----|----|----|----|
|  0 | 3  | 4  | 0  | Med  |

(sec:sqa_training)=
## Training

The goal of this area is to enhance the skills of the developers of the software package.

### Training

#### PR29. Determine project team training needed to fulfill assigned roles and responsibilities.  

Training for developers has not been implemented.  A while back there was a series of presentations that I gave about the Rattlesnake implementation details, but those are very out-of-date now.  I would love to put together a Rattlesnake workshop.

|  Low LOF  |  Med LOF  |  High LOF  | Current Status Grade  | Current Priority  |
|----|----|----|----|----|
|  0 | 3  | 4  | 1  | Low  |

#### PR30. Track training undertaken by project team.

Training is not tracked.

|  Low LOF  |  Med LOF  |  High LOF  | Current Status Grade  | Current Priority  |
|----|----|----|----|----|
|  0 | 3  | 4  | 0  | Low  |

### Summary
This table contains a sum of the values in the above tables to give an overall score of the software quality practices compared to targets.

|  Low LOF  |  Med LOF  |  High LOF  | Current Status Grade  |
|-----|-----|-----|----|
| 32  | 92  | 120 | 28 |

(sec:sqa_scores)=
## Table Definitions

Values in the above tables are defined as follows:

| Rating | Rating Description |
|----|----|
| 5 | Outstanding – the software project team has fully implemented this practice; meaning that a documented process exists for the practice, all team members are fully trained on the process, work products have been produced and managed, and practice plans and results have been shared with all appropriate stakeholders.  |
| 4 | Complete – the software project team has implemented a final (not draft) process for conducting the practice and work products are in place supporting this practice. However, there are still a few activities that need to be addressed (e.g., training, finalizing work products, etc.). Most project team members have been trained in the process implementation. Practice results have been shared with some stakeholders.  |
| 3 | Good – the software project team has partially implemented this practice. For example, a draft of the process for conducting the practice exists or a completed documented process exists with most of the team (but not all) complying with the process. The team has made significant progress in rolling-out an implementation for the process and draft work products that contain significant content exist.  |
| 2 | Fair – the software project team has a preliminary process (e.g., a detailed outline; a well-understood ad hoc team process that is not documented, etc.) for implementing this practice. There may be a preliminary plan about how to proceed with the process and implementation and preliminary work products exist.  |
| 1 | Limited – the software project team has proposed that this practice be implemented and activities and resources for the practice are in the planning stages. It is evident that the project is committed to implementing this practice. At this level, it is typical that resources have not yet been allocated for fulfillment of the practice.  |
| 0 | Absent – the software project team has not yet addressed the implementation of this practice.  |
| NR | Not Reviewed – the software project team (or the appraisal team) determined that this practice should not be reviewed because it is not applicable to the code development environment. A NR determination must include rationale and a waiver from the team. |