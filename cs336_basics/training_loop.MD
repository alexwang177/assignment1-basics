                                                              
   # Training Loop Checklist                                                
                                                                            
   ## Setup                                                                 
   - [X] Choose config mechanism: argparse, dataclass, YAML, or simple      
 constants.                                                                 
   - [ ] Set random seeds for reproducibility.                              
   - [X] Select device: `cpu`, `cuda`, or `mps`.                            
   - [ ] Load train/validation token arrays, preferably with `np.load(...,  
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
   - [ ] Check a dummy batch produces logits of expected shape.             
                                                                            
   ## Optimizer / Schedule                                                  
   - [x] Instantiate AdamW.                                                 
   - [x] Configure learning rate, betas, eps, weight decay.                 
   - [ ] Add cosine LR schedule with warmup.                                
   - [ ] Set optimizer LR each iteration if using schedule.                 
   - [ ] Optionally enable gradient clipping.                               
                                                                            
   ## Training Step                                                         
   - [ ] Sample batch with `get_batch`.                                     
   - [ ] Forward pass.                                                      
   - [ ] Compute next-token cross entropy loss.                             
   - [ ] Zero gradients.                                                    
   - [ ] Backward pass.                                                     
   - [ ] Clip gradients if enabled.                                         
   - [ ] Optimizer step.                                                    
   - [ ] Log training loss periodically.                                    
                                                                            
   ## Validation                                                            
   - [ ] Periodically switch model to eval mode.                            
   - [ ] Run several validation batches under `torch.no_grad()`.            
   - [ ] Average validation loss.                                           
   - [ ] Switch model back to train mode.                                   
   - [ ] Log validation loss.                                               
                                                                            
   ## Checkpointing                                                         
   - [ ] Periodically save checkpoint:                                      
     - [ ] model state                                                      
     - [ ] optimizer state                                                  
     - [ ] iteration                                                        
   - [ ] Add option to resume from checkpoint.                              
   - [ ] Verify resumed training starts from correct iteration.             
                                                                            
   ## Sanity Checks                                                         
   - [ ] Overfit a tiny dataset.                                            
   - [ ] Confirm loss decreases on a small run.                             
   - [ ] Check no NaNs in loss or gradients.                                
   - [ ] Check batch/input/target shapes.                                   
   - [ ] Check model and data are on same device.                           
   - [ ] Confirm checkpoints can be saved and loaded.                       
                                                                            
   ## Experiment Logging                                                    
   - [ ] Print iteration, train loss, validation loss, LR.                  
   - [ ] Optionally log tokens/sec.                                         
   - [ ] Optionally log to WandB or another tracker.                        