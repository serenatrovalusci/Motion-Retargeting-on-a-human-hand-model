using System;
using System.Net.Sockets;
using UnityEngine;

public class HandPoseClient : MonoBehaviour
{
    public HandDataLogger dataLogger;

    private TcpClient client;
    private NetworkStream stream;
    private float[] jointValues = new float[27];

    private Transform[] joints;

    void Start()
    {
        try
        {
            client = new TcpClient("127.0.0.1", 65432);
            stream = client.GetStream();
            Debug.Log(" Connected to Python socket server");
        }
        catch (Exception e)
        {
            Debug.LogError(" Could not connect to Python server: " + e.Message);
        }

        // Map joints in the exact order that matches the neural network output
        joints = new Transform[9] {
            dataLogger.thumb1,
            dataLogger.thumb2,
            dataLogger.thumb3,
            dataLogger.index1,
            dataLogger.index2,
            dataLogger.index3,
            dataLogger.middle1,
            dataLogger.middle2,
            dataLogger.middle3
        };
    }

    void Update()
    {
        if (client == null || !client.Connected || dataLogger == null) return;

        // 1. Send 4 input floats to Python (closure & abduction)
        float[] input = new float[4] {
            dataLogger.thumbClosure,
            dataLogger.indexClosure,
            dataLogger.middleClosure,
            dataLogger.thumbAbductionValue
        };

        byte[] inputBytes = new byte[4 * sizeof(float)];
        Buffer.BlockCopy(input, 0, inputBytes, 0, inputBytes.Length);
        stream.Write(inputBytes, 0, inputBytes.Length);

        // 2. Receive 27 float outputs (9 joints × 3 axes)
        byte[] outputBytes = new byte[27 * sizeof(float)];
        int totalRead = 0;
        while (totalRead < outputBytes.Length)
        {
            int bytesRead = stream.Read(outputBytes, totalRead, outputBytes.Length - totalRead);
            totalRead += bytesRead;
        }

        Buffer.BlockCopy(outputBytes, 0, jointValues, 0, outputBytes.Length);

        // 3. Apply each triplet to the corresponding joint
        for (int i = 0; i < joints.Length; i++)
        {
            if (joints[i] == null) continue;

            int baseIndex = i * 3;
            float x = jointValues[baseIndex];
            float y = jointValues[baseIndex + 1];
            float z = jointValues[baseIndex + 2];

            joints[i].localRotation = Quaternion.Euler(x, y, z);
        }
    }

    void OnApplicationQuit()
    {
        stream?.Close();
        client?.Close();
    }
}
