namespace GameRecommender.Interfaces.Data;

public interface IAppIdMappingHandler
{
    Task AddAppIdToNameMapping(string appId, string name);
}