using NUnit.Framework;
using Moq;
using GameRecommender.Services;
using NUnit.Framework.Legacy;

namespace GameRecommender.Tests;
[TestFixture]
public class SteamGameFetcherTests
{
    private SteamGameFetcher _steamGameFetcher;

    [SetUp]
    public void Setup()
    {
    }

    [Test]
    public async Task CheckCorrectFetching()
    {
        string steamIdToTest = "76561198241827750";
        var gamesPlayed = await SteamGameFetcher.GetSteamGamesFromXmlAsync(steamIdToTest);
        foreach (var data in gamesPlayed)
        {
            Console.WriteLine($"{data.Name} {data.AppId} {data.HoursOnRecord}");
        }
        ClassicAssert.IsTrue(gamesPlayed.Exists(b => b.Name == "Factorio"));
    }
}
