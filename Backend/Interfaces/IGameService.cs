namespace GameRecommender.Interfaces;

public interface IGameService
{
    public Task AddOpinionForUserAndGame(Guid userId, string appId, bool opinion);
    public Task AddAppIdToNameMapping(string appId, string name);
}