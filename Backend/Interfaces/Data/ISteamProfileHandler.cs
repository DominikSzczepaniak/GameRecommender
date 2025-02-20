namespace GameRecommender.Interfaces.Data;

public interface ISteamProfileHandler
{
    Task SetUserSteamProfileId(Guid userId, string steamProfileId);
    Task<string> GetUserSteamId(Guid userId);
}