using GameRecommender.Interfaces.Data;

namespace GameRecommender.Data;

public class PostgresSteamProfileHandler(PostgresConnectionPool connectionPool) : PostgresBase(connectionPool), ISteamProfileHandler
{
    public async Task SetUserSteamProfileId(Guid userId, string steamProfileId)
    {
        using var connection = await GetConnectionAsync();
        var command = connection.CreateCommand();
        command.CommandText = @"
            INSERT INTO UserToSteamIds (UserId, SteamId)
            VALUES (@UserId, @SteamId)
            ON CONFLICT (UserId) DO UPDATE SET SteamId = EXCLUDED.SteamId";
        command.Parameters.Add(new Npgsql.NpgsqlParameter("@UserId", userId));
        command.Parameters.Add(new Npgsql.NpgsqlParameter("@SteamId", steamProfileId));

        await command.ExecuteNonQueryAsync();
    }

    public async Task<string> GetUserSteamId(Guid userId)
    {
        using var connection = await GetConnectionAsync();
        var command = connection.CreateCommand();
        command.CommandText = "SELECT SteamId FROM UserToSteamIds WHERE UserId = @UserId";
        command.Parameters.Add(new Npgsql.NpgsqlParameter("@UserId", userId));

        var result = await command.ExecuteScalarAsync();
        if (result == null)
        {
            throw new ArgumentException("User doesn't exist or SteamID not saved");
        }

        return result.ToString();
    }
}