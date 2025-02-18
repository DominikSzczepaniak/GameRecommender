using System.Text;
using System.Text.Json;
using GameRecommender.Data;
using GameRecommender.Interfaces;
using GameRecommender.Models;

namespace GameRecommender.Services;

public class RecommenderApiService : IRecommenderApiService
{
    private readonly HttpClient _httpClient;
    private readonly IDatabaseHandler _databaseHandler;

    public RecommenderApiService(HttpClient httpClient, IDatabaseHandler databaseHandler)
    {
        _httpClient = httpClient;
        _databaseHandler = databaseHandler;
    }

    public async Task<List<string>> GetGameList(string userId, int k, string host)
    {
        var requestData = new { userId, k };
        var jsonContent = new StringContent(JsonSerializer.Serialize(requestData), Encoding.UTF8, "application/json");
        var response = await _httpClient.PostAsync($"{host}/recommendations", jsonContent);
        response.EnsureSuccessStatusCode();
        var responseString = await response.Content.ReadAsStringAsync();
        return JsonSerializer.Deserialize<List<string>>(responseString) ?? new List<string>();
    }

    public async Task<bool> LearnUser(List<UserGameDao> gameList, string host)
    {
        var jsonContent = new StringContent(JsonSerializer.Serialize(gameList), Encoding.UTF8, "application/json");
        var response = await _httpClient.PostAsync($"{host}/learnUser", jsonContent);
        if (!response.IsSuccessStatusCode)
            return false;
        return true;
    }
}