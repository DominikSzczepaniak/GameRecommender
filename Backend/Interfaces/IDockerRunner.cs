namespace GameRecommender.Interfaces;

public interface IDockerRunner
{
    public Task<List<String>> GetRecommendations(Guid userId, int numberOfRecommendations, int engineNumber);
}