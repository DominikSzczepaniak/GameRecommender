namespace GameRecommender.Models;

public class UserGameDto
{
    public string AppId { get; private set; }
    public bool Opinion { get; private set; }

    public UserGameDto(string appId, bool opinion)
    {
        AppId = appId;
        Opinion = opinion;
    }

    public UserGameLogic ToLogic()
    {
        return new UserGameLogic(AppId, null, Opinion);
    }
}