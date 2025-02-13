namespace GameRecommender.Data;

public class GameData
{
    public string AppId { get; set; }
    public string Name { get; set; }
    public double HoursOnRecord { get; set; }

    public GameData(string appId, string name, double hoursOnRecord)
    {
        AppId = appId;
        Name = name;
        HoursOnRecord = hoursOnRecord;
    }
}