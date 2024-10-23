def writing_logs(writer, train_metrics, val_metrics, epoch):
    """-------------------------Loss--------------------------------------------------------"""               
    writer.add_scalars('Loss/mean',
                        {'train':train_metrics['Loss'][3],
                        'Val':val_metrics['Loss'][3]},
                        epoch)
                        
    writer.add_scalars('Loss/OM',
                        {'train':train_metrics['Loss'][0],
                            'Val':val_metrics['Loss'][0]},
                            epoch)
    
    writer.add_scalars('Loss/OP',
                        {'train':train_metrics['Loss'][1],
                            'Val':val_metrics['Loss'][1]},
                            epoch)
    
    writer.add_scalars('Loss/IOP',
                        {'train':train_metrics['Loss'][2],
                            'Val':val_metrics['Loss'][2]},
                            epoch)
                    
    """-------------------------Dice--------------------------------------------------------"""
    writer.add_scalars('Dice/mean',
                        {'train':train_metrics['Dice'][3],
                            'Val':val_metrics['Dice'][3]},
                            epoch)

    writer.add_scalars('Dice/OM',
                        {'train':train_metrics['Dice'][0],
                            'Val':val_metrics['Dice'][0]},
                            epoch)
    
    writer.add_scalars('Dice/OP',
                        {'train':train_metrics['Dice'][1],
                            'Val':val_metrics['Dice'][1]},
                            epoch)
    
    writer.add_scalars('Dice/IOP',
                        {'train':train_metrics['Dice'][2],
                            'Val':val_metrics['Dice'][2]},
                            epoch)
    
    """-------------------------Precision--------------------------------------------------------"""
    writer.add_scalars('Precision/mean',
                        {'train':train_metrics['Precision'][3],
                            'Val':val_metrics['Precision'][3]},
                            epoch)

    writer.add_scalars('Precision/OM',
                        {'train':train_metrics['Precision'][0],
                            'Val':val_metrics['Precision'][0]},
                            epoch)
    
    writer.add_scalars('Precision/OP',
                        {'train':train_metrics['Precision'][1],
                            'Val':val_metrics['Precision'][1]},
                            epoch)
    
    writer.add_scalars('Precision/IOP',
                        {'train':train_metrics['Precision'][2],
                            'Val':val_metrics['Precision'][2]},
                            epoch)
    
    """-------------------------Recall--------------------------------------------------------"""
    writer.add_scalars('Recall/mean',
                        {'train':train_metrics['Recall'][3],
                            'Val':val_metrics['Recall'][3]},
                            epoch)
    
    writer.add_scalars('Recall/OM',
                        {'train':train_metrics['Recall'][0],
                            'Val':val_metrics['Recall'][0]},
                            epoch)
    
    writer.add_scalars('Recall/OP',
                        {'train':train_metrics['Recall'][1],
                            'Val':val_metrics['Recall'][1]},
                            epoch)
    
    writer.add_scalars('Recall/IOP',
                        {'train':train_metrics['Recall'][2],
                            'Val':val_metrics['Recall'][2]},
                            epoch)
    
    """-------------------------F1_scores--------------------------------------------------------"""
    writer.add_scalars('F1_scores/mean',
                        {'train':train_metrics['F1_scores'][3],
                            'Val':val_metrics['F1_scores'][3]},
                            epoch)
    
    writer.add_scalars('F1_scores/OM',
                        {'train':train_metrics['F1_scores'][0],
                            'Val':val_metrics['F1_scores'][0]},
                            epoch)
    
    writer.add_scalars('F1_scores/OP',
                        {'train':train_metrics['F1_scores'][1],
                            'Val':val_metrics['F1_scores'][1]},
                            epoch)
    
    writer.add_scalars('F1_scores/IOP',
                        {'train':train_metrics['F1_scores'][2],
                            'Val':val_metrics['F1_scores'][2]},
                            epoch)
    
    """-------------------------mIoU--------------------------------------------------------"""
    writer.add_scalars('mIoU/mean',
                        {'train':train_metrics['mIoU'][3],
                            'Val':val_metrics['mIoU'][3]},
                            epoch)
    
    writer.add_scalars('mIoU/OM',
                        {'train':train_metrics['mIoU'][0],
                            'Val':val_metrics['mIoU'][0]},
                            epoch)
    
    writer.add_scalars('mIoU/OP',
                        {'train':train_metrics['mIoU'][1],
                            'Val':val_metrics['mIoU'][1]},
                            epoch)
    
    writer.add_scalars('mIoU/IOP',
                        {'train':train_metrics['mIoU'][2],
                            'Val':val_metrics['mIoU'][2]},
                            epoch)
    
    """-------------------------Accuracy--------------------------------------------------------"""
    writer.add_scalars('Accuracy/mean',
                        {'train':train_metrics['Accuracy'][3],
                            'Val':val_metrics['Accuracy'][3]},
                            epoch)
    
    writer.add_scalars('Accuracy/OM',
                        {'train':train_metrics['Accuracy'][0],
                            'Val':val_metrics['Accuracy'][0]},
                            epoch)
    
    writer.add_scalars('Accuracy/OP',
                        {'train':train_metrics['Accuracy'][1],
                            'Val':val_metrics['Accuracy'][1]},
                            epoch)
    
    writer.add_scalars('Accuracy/IOP',
                        {'train':train_metrics['Accuracy'][2],
                            'Val':val_metrics['Accuracy'][2]},
                            epoch)
            