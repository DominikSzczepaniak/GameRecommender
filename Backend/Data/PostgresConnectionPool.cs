using System.Collections.Concurrent;
using Npgsql;

namespace Data;

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