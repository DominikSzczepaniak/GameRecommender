using System.Collections.Concurrent;
using System.Data;
using Npgsql;
namespace GameRecommender.Data;

public class PostgresConnectionPool
{
    private readonly string _connectionString;
    private readonly ConcurrentBag<NpgsqlConnection> _connectionPool;
    private readonly int _maxPoolSize;
    public PostgresConnectionPool(string connectionString, int maxPoolSize)
    {
        _connectionString = connectionString;
        _maxPoolSize = maxPoolSize;
        _connectionPool = new ConcurrentBag<NpgsqlConnection>();
        InitializePool();
    }
    private void InitializePool()
    {
        for (int i = 0; i < _maxPoolSize; i++)
        {
            var connection = new NpgsqlConnection(_connectionString);
            connection.Open();
            _connectionPool.Add(connection);
        }
    }
    public async Task<NpgsqlConnection> GetConnectionAsync()
    {
        if (_connectionPool.TryTake(out var connection))
        {
            if (connection.State == ConnectionState.Closed)
            {
                await connection.OpenAsync();
            }
            return connection;
        }
        var newConnection = new NpgsqlConnection(_connectionString);
        await newConnection.OpenAsync();
        return newConnection;
    }
    public void Dispose()
    {
        foreach (var connection in _connectionPool)
        {
            connection.Dispose();
        }
    }
}