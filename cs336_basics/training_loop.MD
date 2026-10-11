                                                              
   # Training Loop Checklist                                                
                                                                            
   ## Setup                                                                 
   - [X] Choose config mechanism: argparse, dataclass, YAML, or simple      
 constants.                                                                 
   - [x] Set random seeds for reproducibility.                              
   - [X] Select device: `cpu`, `cuda`, or `mps`.                            
   - [x] Load train/validation token arrays, preferably with `np.load(...,  
 mmap_mode="r")`.                                                           
                                                                            
   ## Model                                                                 
   - [x] Instantiate model with configurable hyperparameters:               
     - [x] vocab size                                                       
     - [x] context length                                                   
     - [x] number of layers                                                 
     - [x] number of heads                                                  
     - [x] embedding dimension                                              
     - [x] FFN dimension, if applicable                                     
   - [x] Move model to device.                                              
   - [x] Check a dummy batch produces logits of expected shape.             
                                                                            
   ## Optimizer / Schedule                                                  
   - [x] Instantiate AdamW.                                                 
   - [x] Configure learning rate, betas, eps, weight decay.                 
   - [x] Add cosine LR schedule with warmup.                                
   - [x] Set optimizer LR each iteration if using schedule.                 
   - [x] Optionally enable gradient clipping.                               
                                                                            
   ## Training Step                                                         
   - [x] Sample batch with `get_batch`.                                     
   - [x] Forward pass.                                                      
   - [x] Compute next-token cross entropy loss.                             
   - [x] Zero gradients.                                                    
   - [x] Backward pass.                                                     
   - [x] Clip gradients if enabled.                                         
   - [x] Optimizer step.                                                    
   - [x] Log training loss periodically.                                    
                                                                            
   ## Validation                                                            
   - [ ] Periodically switch model to eval mode.                            
   - [ ] Run several validation batches under `torch.no_grad()`.            
   - [ ] Average validation loss.                                           
   - [ ] Switch model back to train mode.                                   
   - [ ] Log validation loss.                                               
                                                                            
   ## Checkpointing                                                         
   - [x] Periodically save checkpoint:                                      
     - [x] model state                                                      
     - [x] optimizer state                                                  
     - [x] iteration                                                        
   - [x] Add option to resume from checkpoint.                              
   - [x] Verify resumed training starts from correct iteration.             
                                                                            
   ## Sanity Checks                                                         
   - [x] Overfit a tiny dataset.                                            
   - [x] Confirm loss decreases on a small run.                             
   - [x] Check no NaNs in loss or gradients.                                
   - [x] Check batch/input/target shapes.                                   
   - [x] Check model and data are on same device.                           
   - [x] Confirm checkpoints can be saved and loaded.                       
                                                                            
   ## Experiment Logging                                                    
   - [x] Print iteration, train loss, validation loss, LR.                  
   - [ ] Optionally log tokens/sec.                                         
   - [ ] Optionally log to WandB or another tracker.                        