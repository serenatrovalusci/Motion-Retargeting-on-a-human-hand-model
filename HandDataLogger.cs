using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class HandDataLogger : MonoBehaviour
{
    private string filePath;
    private StreamWriter writer;
    
    // Riferimento ai joint di Weart
    public Transform thumb1;
    public Transform thumb2;
    public Transform thumb3;

    public Transform index1;
    public Transform index2;
    public Transform index3;

    public Transform middle1;
    public Transform middle2;
    public Transform middle3;

    public float closure;

    // Start is called before the first frame update
    void Start()
    {
        filePath = Application.persistentDataPath + "/hand_dataset.csv";
        
        bool fileExists = File.Exists(filePath);  // Controllo PRIMA se il file esiste

        // Apro il file solo DOPO aver controllato se esiste
        writer = new StreamWriter(filePath, true);

        if (!fileExists)
        {
            writer.WriteLine("Joint1T,Joint2T,Joint3T,Joint1I,Joint2I,Joint3I,Joint1M,Joint2M,Joint3M,Closure");
            writer.Flush();
        }

        // Trova i joint nel prefab usando i nomi precisi
        thumb1 = transform.Find("DEF-thumb.01.R");
        thumb2 = transform.Find("DEF-thumb.02.R");
        thumb3 = transform.Find("DEF-thumb.03.R");

        index1 = transform.Find("DEF-f_index.01.R");
        index2 = transform.Find("DEF-f_index.02.R");
        index3 = transform.Find("DEF-f_index.03.R");

        middle1 = transform.Find("DEF-f_middle.01.R");
        middle2 = transform.Find("DEF-f_middle.02.R");
        middle3 = transform.Find("DEF-f_middle.03.R");

        // Controllo se tutti i riferimenti ai joint sono stati trovati
        if (thumb1 == null || thumb2 == null || thumb3 == null ||
            index1 == null || index2 == null || index3 == null ||
            middle1 == null || middle2 == null || middle3 == null)
        {
            Debug.LogError(" Error: Some joints have not been found. Check joint names in the Hierarchy!");
        }
        else
        {
            Debug.Log(" All joints successfully found!");
        }
    }

    void Update()
    {
        if (thumb1 != null && thumb2 != null && thumb3 != null &&
            index1 != null && index2 != null && index3 != null &&
            middle1 != null && middle2 != null && middle3 != null)
        {
            // Ottieni le rotazioni dei joint
            float Joint1T = thumb1.localRotation.eulerAngles.x;
            float Joint2T = thumb2.localRotation.eulerAngles.x;
            float Joint3T = thumb3.localRotation.eulerAngles.x;

            float Joint1I = index1.localRotation.eulerAngles.x;
            float Joint2I = index2.localRotation.eulerAngles.x;
            float Joint3I = index3.localRotation.eulerAngles.x;

            float Joint1M = middle1.localRotation.eulerAngles.x;
            float Joint2M = middle2.localRotation.eulerAngles.x;
            float Joint3M = middle3.localRotation.eulerAngles.x;

            // Registra i dati solo se `writer` non è null
            if (writer != null)
            {
                LogData(Joint1T, Joint2T, Joint3T, Joint1I, Joint2I, Joint3I, Joint1M, Joint2M, Joint3M, closure);
            }
        }
    }

    public void LogData(float Joint1T, float Joint2T, float Joint3T, float Joint1I, float Joint2I, float Joint3I, float Joint1M, float Joint2M, float Joint3M, float closure)
    {
        if (writer != null)  // Controllo che `writer` sia valido
        {
            string line = Joint1T + "," + Joint2T + "," + Joint3T + "," + Joint1I + "," + Joint2I + "," + Joint3I + "," + Joint1M + "," + Joint2M + "," + Joint3M + "," + closure;
            writer.WriteLine(line);
            writer.Flush();
        }
        else
        {
            Debug.LogError(" Error: writer is null, data not logged!");
        }
    }

    void OnApplicationQuit()
    {
        if (writer != null)
        {
            writer.Close();
        }
    }
}
