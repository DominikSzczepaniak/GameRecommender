namespace GameRecommender.Interfaces;

public interface IGameLibrary
{
    public Task SetUserSteamProfile(Guid userId, string steamProfileLink);
}