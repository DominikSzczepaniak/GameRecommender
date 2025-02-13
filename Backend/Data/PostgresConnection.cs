using GameRecommender.Data;
using GameRecommender.Models;
using Npgsql;

namespace Data;

public class PostgresConnection(PostgresConnectionPool connectionPool) : IDatabaseHandler, IDisposable
{
    public void Dispose()
    {
        connectionPool.Dispose();
    }

    private async Task<NpgsqlConnection> GetConnectionAsync()
    {
        return await connectionPool.GetConnectionAsync();
    }

    public Task<User> RegisterUser(User user)
    {
        throw new NotImplementedException();
    }

    public Task<User> LoginByUsername(string username, string password)
    {
        throw new NotImplementedException();
    }

    public Task<User> UpdateUser(User user)
    {
        throw new NotImplementedException();
    }

    public Task<bool> DeleteUser(User user)
    {
        throw new NotImplementedException();
    }

    public Task SetUserSteamProfileId(int userId, string steamProfileId)
    {
        throw new NotImplementedException();
    }
}