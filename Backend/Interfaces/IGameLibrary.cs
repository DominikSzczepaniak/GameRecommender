namespace GameRecommender.Interfaces;

public interface IGameLibrary
{
    public Task SetUserSteamProfile(int userId, string steamProfileLink);
}