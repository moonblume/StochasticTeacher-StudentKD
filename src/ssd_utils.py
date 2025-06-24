import torch
import torch.nn.functional as F

def cosine_similarity_loss_all_reps(student_rep, teacher_reps):
    """
    Compute the loss to maximize cosine similarity between student and teacher reps.
    Used for Ablation when distilling from all stochastic teacher representations 
    
    Args:
        student_rep (torch.Tensor): (batch_size, feature_dim) - Student representation.
        teacher_reps (torch.Tensor): (batch_size, num_teacher_reps, feature_dim) - Teacher representations.
    
    Returns:
        torch.Tensor: Cosine similarity loss.
    """

    teacher_reps = torch.stack(teacher_reps, dim=1)  # (batch_size, num_teacher_reps, feature_dim)
    student_rep = student_rep.unsqueeze(1)

    cos_sim = F.cosine_similarity(student_rep, teacher_reps, dim=2)  # (batch_size, num_teacher_reps)

    mean_cos_sim = cos_sim.mean(dim=1)  # (batch_size,)

    loss = 1 - mean_cos_sim.mean()  # Mean over batch
    return loss


def calculate_teacher_student_loss_batch(student_rep, teacher_reps, top_k):
    """
    Calculate a loss value for a batch of student representations and teacher representations.

    Args:
        student_reps (torch.Tensor): Batch of student feature representations, shape (B, D).
        teacher_reps (list[torch.Tensor]): List of teacher feature representations, each of shape (D,).
        top_k (int): Number of closest teacher representations to include in the loss.

    Returns:
        torch.Tensor: The calculated loss value for the entire batch.
    """
    # Convert teacher_reps list to a tensor of shape (N, D), where N is the number of teachers
    teacher_reps_tensor = torch.stack(teacher_reps)  # Shape: (N, D)
    #permute to batch first
    teacher_reps_tensor = teacher_reps_tensor.permute(1,0,2)
    student_rep= student_rep.unsqueeze(1)

    # Convert cosine similarity to cosine distance
    # cosine_similarities = F.cosine_similarity(student_rep, teacher_reps_tensor, dim=2)


   
    #calculate the dot product between the student and teacher reps
    dot_product = torch.sum(student_rep*teacher_reps_tensor, dim=2)

    #calculate attention weights without temperature scaling
    # attention_weights = F.softmax(dot_product, dim=1)

    #temperature scaled attention weights 
    attention_weights = F.softmax(dot_product/5, dim=1)
    

    #normalize the attention weights to sum to 1
    attention_weights = attention_weights/attention_weights.sum(dim=1, keepdim=True)


    # select attention weights that are in the 90th percentile 
    attention_weights = select_by_dynamic_threshold(attention_weights, percentile=90)



    #calculate the attention weighted teacher reps 
    attended_teacher_reps = torch.sum(attention_weights.unsqueeze(-1)*teacher_reps_tensor, dim=1)

    # Calculate the loss as the mean squared error between the student and attended teacher representations
    loss = F.mse_loss(student_rep.squeeze(1), attended_teacher_reps, reduction='mean')


    return loss



def select_by_dynamic_threshold(weights, percentile=90):
    """
    Select attention weights adaptively based on a dynamic threshold.

    Args:
        weights (torch.Tensor): Tensor of attention weights (batch_size, num_weights).
        percentile (float): Percentile threshold to determine selection.

    Returns:
        torch.Tensor: Tensor with weights above the adaptive threshold retained.
    """
    thresholds = torch.quantile(weights, percentile / 100.0, dim=-1, keepdim=True)
    selected_weights = torch.where(weights >= thresholds, weights, torch.zeros_like(weights))
    return selected_weights / torch.sum(selected_weights, dim=-1, keepdim=True)
