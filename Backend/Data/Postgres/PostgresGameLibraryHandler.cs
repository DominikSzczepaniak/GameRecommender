using GameRecommender.Interfaces.Data;
using GameRecommender.Models;

namespace GameRecommender.Data;

public class PostgresGameLibraryHandler(PostgresConnectionPool connectionPool) : PostgresBase(connectionPool), IGameLibraryHandler
{
    public async Task AddGameToUserLibrary(UserGameDao userGameDao)
    {
        using var connection = await GetConnectionAsync();
        var command = connection.CreateCommand();
        command.CommandText = "INSERT INTO UserGames (UserId, AppId, Playtime, Opinion) VALUES (@UserId, @AppId, @Playtime, @Opinion)";
        command.Parameters.Add(new Npgsql.NpgsqlParameter("@UserId", userGameDao.UserId));
        command.Parameters.Add(new Npgsql.NpgsqlParameter("@AppId", userGameDao.AppId));
        command.Parameters.Add(new Npgsql.NpgsqlParameter("@Playtime", userGameDao.Playtime));
        command.Parameters.Add(new Npgsql.NpgsqlParameter("@Opinion", userGameDao.Opinion));

        await command.ExecuteNonQueryAsync();
    }

    public async Task<List<UserGameLogic>> GetUserGames(Guid userId)
    {
        using var connection = await GetConnectionAsync();
        var command = connection.CreateCommand();
        command.CommandText = @"
            SELECT ug.AppId, ug.Playtime, ug.Opinion, an.Name 
            FROM UserGames ug
            LEFT JOIN AppIdToNames an ON ug.AppId = an.AppId
            WHERE ug.UserId = @UserId";
        command.Parameters.Add(new Npgsql.NpgsqlParameter("@UserId", userId));

        var games = new List<UserGameLogic>();
        using var reader = await command.ExecuteReaderAsync();
        while (await reader.ReadAsync())
        {
            games.Add(new UserGameLogic(
                reader.GetString(reader.GetOrdinal("AppId")),
                reader.GetInt32(reader.GetOrdinal("Playtime")),
                reader.GetBoolean(reader.GetOrdinal("Opinion")),
                reader.IsDBNull(reader.GetOrdinal("Name")) ? "" : reader.GetString(reader.GetOrdinal("Name"))
            ));
        }

        return games;
    }

    public async Task AddOpinionForUserAndGame(UserGameDao userGameDao)
    {
        using var connection = await GetConnectionAsync();
        var command = connection.CreateCommand();
        command.CommandText = @"
            UPDATE UserGames 
            SET Opinion = @Opinion 
            WHERE UserId = @UserId AND AppId = @AppId";
        command.Parameters.Add(new Npgsql.NpgsqlParameter("@UserId", userGameDao.UserId));
        command.Parameters.Add(new Npgsql.NpgsqlParameter("@AppId", userGameDao.AppId));
        command.Parameters.Add(new Npgsql.NpgsqlParameter("@Opinion", userGameDao.Opinion));

        var rowsAffected = await command.ExecuteNonQueryAsync();
        if (rowsAffected == 0)
        {
            await AddGameToUserLibrary(userGameDao);
        }
    }
}