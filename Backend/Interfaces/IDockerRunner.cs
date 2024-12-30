namespace GameRecommender.Interfaces;

public interface IDockerRunner
{
    public Task<List<String>> GetRecommendations(int userId, int engineNumber);
}