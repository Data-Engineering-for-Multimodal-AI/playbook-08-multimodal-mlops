# Pipeline Architecture Diagrams

This directory contains architecture diagrams for the MLOps pipeline.

## MLOps Pipeline Architecture

Our pipeline architecture follows a closed-loop MLOps workflow that emphasizes continuous monitoring and automated retraining.

The `pipeline_architecture.md` file contains a Mermaid diagram that visualizes the end-to-end flow of our MLOps pipeline with the following components:

1. **Data Ingestion** (Light Blue): Processes raw INS and video data from the Oxford RobotCar dataset
2. **Feature Engineering** (Light Green): Extracts meaningful features using sliding windows
3. **Model Training** (Light Purple): Trains and validates ML models with aligned multimodal data
4. **Model Registry** (Light Purple): Stores and versions trained models
5. **Model Serving** (Light Yellow): Provides predictions through a REST API
6. **Drift Detection** (Light Orange): Monitors for data distribution changes over time
7. **Automated Retraining** (Light Orange): Triggers new model training based on drift signals

The diagram highlights the closed-loop nature of our architecture: 
- Data flows from ingestion through feature engineering to model training
- Trained models are registered and deployed to the serving layer
- The serving layer generates predictions which are monitored for drift
- When significant drift is detected, automated retraining is triggered
- Newly trained models are registered and replace the serving model if they perform better

This architecture ensures our models remain accurate even as the underlying data distributions evolve over time, implementing true ML automation with human oversight.

### Viewing the Diagram

GitHub automatically renders Mermaid diagrams in markdown files. To view the diagram:
1. Open `pipeline_architecture.md` directly in GitHub
2. The diagram will be rendered automatically in the GitHub interface

Alternatively, you can use any Mermaid-compatible viewer:
- [Mermaid Live Editor](https://mermaid-js.github.io/mermaid-live-editor/)
- VS Code with the Mermaid extension
- Obsidian, which natively supports Mermaid 