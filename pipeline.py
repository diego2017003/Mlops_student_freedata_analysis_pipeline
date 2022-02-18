'''
Reyne Jasson
17 feb. 2022

This script controls one end-to-end pipeline with clearML
'''

from clearml import Task
from clearml.automation import PipelineController


pipe = PipelineController(
     name='A student Pipeline',
     project='a ML example',
     version='0.0.1',
     add_pipeline_tags=False,
)

pipe.add_step(name='stage_data', base_task_project='a ML example', 
            base_task_name='Download Raw data',base_task_id="4eb7af98b1d94e01a0314db4426bcae3")

pipe.add_step(name = "Pre Processes Data",
     parents=['stage_data'],
     base_task_project='a ML example',
     base_task_name='Prepreocessing',base_task_id="71845909e9b643fca92e5902c32265a1")

pipe.add_step(name="Train model",
                parents=["Pre Processes Data"],
                base_task_project='a ML example',
            base_task_name="logist training",
            base_task_id="cc2edd8da8784a2c8ceeb35b82e4b72f")

pipe.start_locally(run_pipeline_steps_locally=True)

print('done')



