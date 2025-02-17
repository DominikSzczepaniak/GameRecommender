namespace GameRecommender.Models;

public class AddOpinionRequest
{
    public User user { get; set; }
    public UserGameDto[] gameDto { get; set; }
}